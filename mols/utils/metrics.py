import torch
import pickle
import gzip
import torch.nn as nn
import numpy as np
import networkx as nx

from mol_mdp_ext import MolMDPExtended


def get_mol_path_graph(mol, bpath):
    mdp = MolMDPExtended(bpath)
    mdp.post_init(torch.device('cpu'), 'block_graph')
    mdp.build_translation_table()
    mdp.floatX = torch.float
    agraph = nx.DiGraph()
    agraph.add_node(0)
    ancestors = [mol]
    ancestor_graphs = []

    par = mdp.parents(mol)
    mstack = [i[0] for i in par]
    pstack = [[0, a] for i,a in par]
    while len(mstack):
        m = mstack.pop() #pop = last item is default index
        p, pa = pstack.pop()
        match = False
        mgraph = mdp.get_nx_graph(m)
        for ai, a in enumerate(ancestor_graphs):
            if mdp.graphs_are_isomorphic(mgraph, a):
                agraph.add_edge(p, ai+1, action=pa)
                match = True
                break
        if not match:
            agraph.add_edge(p, len(ancestors), action=pa) #I assume the original molecule = 0, 1st ancestor = 1st parent = 1
            ancestors.append(m) #so now len(ancestors) will be 2 --> and the next edge will be to the ancestor labelled 2
            ancestor_graphs.append(mgraph)
            if len(m.blocks):
                par = mdp.parents(m)
                mstack += [i[0] for i in par]
                pstack += [(len(ancestors)-1, i[1]) for i in par]

    for u, v in agraph.edges:
        c = mdp.add_block_to(ancestors[v], *agraph.edges[(u,v)]['action'])
        geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                        mdp.get_nx_graph(ancestors[u], true_block=True))
        if not geq: # try to fix the action
            block, stem = agraph.edges[(u,v)]['action']
            for i in range(len(ancestors[v].stems)):
                c = mdp.add_block_to(ancestors[v], block, i)
                geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                                mdp.get_nx_graph(ancestors[u], true_block=True))
                if geq:
                    agraph.edges[(u,v)]['action'] = (block, i)
                    break
        if not geq:
            raise ValueError('could not fix action')
    for u in agraph.nodes:
        agraph.nodes[u]['mol'] = ancestors[u]
    return agraph
    

def compute_correlation(model, mdp, bpath, test_mols_path, floatX=torch.float64, min_blocks=2, entropy_coeff=1):
    device = torch.device('cuda')
    tf = lambda x: torch.tensor(x, device=device).to(floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    test_mols = pickle.load(gzip.open(test_mols_path))
    logsoftmax = nn.LogSoftmax(0)
    logp = []
    reward = []
    numblocks = []
    for moli in (test_mols[:1000]):
        reward.append(np.log(moli[0]))
        try:
            agraph = get_mol_path_graph(moli[1], bpath)
        except:
            continue
        s = mdp.mols2batch([mdp.mol2repr(agraph.nodes[i]['mol']) for i in agraph.nodes])
        numblocks.append(len(moli[1].blocks))
        with torch.no_grad():
            stem_out_s, mol_out_s = model(s, None)  # get the mols_out_s for ALL molecules not just the end one.
            # Application of entropy coefficient
            stem_out_s = stem_out_s / entropy_coeff
            mol_out_s = mol_out_s / entropy_coeff
        per_mol_out = []
        # Compute pi(a|s)
        for j in range(len(agraph.nodes)):
            a,b = s._slice_dict['stems'][j:j+2]

            stop_allowed = len(agraph.nodes[j]['mol'].blocks) >= min_blocks
            mp = entropy_coeff * logsoftmax(torch.cat([
                stem_out_s[a:b].reshape(-1),
                # If num_blocks < min_blocks, the model is not allowed to stop
                mol_out_s[j, :1] if stop_allowed else tf([-1000])]))
            per_mol_out.append((mp[:-1].reshape((-1, stem_out_s.shape[1])), mp[-1]))

        # When the model reaches 8 blocks, it is stopped automatically. If instead it stops before
        # that, we need to take into account the STOP action's logprob
        if len(moli[1].blocks) < 8:
            stem_out_last, mol_out_last = model(mdp.mols2batch([mdp.mol2repr(moli[1])]), None)
            # Application of entropy coefficient
            stem_out_last = stem_out_last / entropy_coeff
            mol_out_last = mol_out_last / entropy_coeff
            mplast = entropy_coeff * logsoftmax(torch.cat([stem_out_last.reshape(-1), mol_out_last[0, :1]]))
            MSTOP = mplast[-1]

        # assign logprob to edges
        for u,v in agraph.edges:
            a = agraph.edges[u,v]['action']
            if a[0] == -1:
                agraph.edges[u,v]['logprob'] = per_mol_out[v][1]
            else:
                agraph.edges[u,v]['logprob'] = per_mol_out[v][0][a[1], a[0]]

        # propagate logprobs through the graph
        for n in list(nx.topological_sort(agraph))[::-1]: 
            for c in agraph.predecessors(n): 
                if len(moli[1].blocks) < 8 and c == 0:
                    agraph.nodes[c]['logprob'] = torch.logaddexp(
                        agraph.nodes[c].get('logprob', tf(-1000)),
                        agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob', 0) + MSTOP)
                else:
                    agraph.nodes[c]['logprob'] = torch.logaddexp(
                        agraph.nodes[c].get('logprob', tf(-1000)),
                        agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob',0))

        logp.append((moli, agraph.nodes[n]['logprob'].item())) #add the first item
    return logp

