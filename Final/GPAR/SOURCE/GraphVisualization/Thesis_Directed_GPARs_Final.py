import time
import re
import math
import queue
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MODIFY_INDEX = 0
VACANT_INDEX = -1
VACANT_LABEL = -1

## Class Edge
class MyEdge(tuple):
  def __repr__(self):
    return 'e {} {} {}'.format(self[0], self[1], self[2])

## Class MyNode

class MyNode:
  def __init__(self, id = VACANT_INDEX, label = VACANT_LABEL, to_nodes = set(), from_nodes = set()):
    self.id = id
    self.to_nodes = to_nodes.copy()   ## list id_nodes that from MyNode
    self.from_nodes = from_nodes.copy() ## list id_nodes that to MyNode
    self.label = label 

  def __repr__(self):
    return 'v {} {}'.format(self.id, self.label)

## Class MyGraph

class MyGraph:
  def __init__(self, nodes=[], edges=set()):
    self.nodes = nodes.copy()
    self.edges = edges.copy()


  def loadNode (self, filename):  
    count = 0
    with open(filename, 'r') as f:
      for line in f.readlines():
        count += 1
    self.nodes = [None]*count     ## load node id resp index self.nodes 
    with open(filename, 'r') as f:
      for line in f.readlines():
        line = line.strip('\n')
        line = re.split(r'\D+', line)
        self.nodes[int(line[0])+MODIFY_INDEX] = MyNode(int(line[0])+MODIFY_INDEX, int(line[1]))

    
  def loadEdge (self, filename):
    with open(filename, 'r') as f:
      for line in f.readlines():
        line = line.strip('\n')
        line = re.split(r'\D+', line)
        for i in range(0,2):
          line[i] = int(line[i]) + MODIFY_INDEX
        #if len(line) > 2:    ### edge's label - just ignore
         # line[2] = int(line[2])
          #self.edges.append(MyEdge(line[0], line[1], line[2]))
        #else:
        self.edges.add(MyEdge((line[0], line[1], VACANT_LABEL)))   #####
        self.nodes[line[0]].to_nodes.add(line[1])
        self.nodes[line[1]].from_nodes.add(line[0])
  

  def loadData(self, f1, f2):
    self.loadNode(f1)
    self.loadEdge(f2)

  def addNode(self, node):
    self.nodes.append(node)

  def addEdge(self, edge):
    self.edges.add(edge)
    self.nodes[edge[0]].to_nodes.add(edge[1])
    self.nodes[edge[1]].from_nodes.add(edge[0])
  
  def getNumEdges(self):
    return len(self.edges)

  def getDFSLabels(self):
    res = set()
    for e in self.edges:
      vlb1 = self.nodes[e[0]].label
      vlb2 = self.nodes[e[1]].label
      if (vlb1, e[2], vlb2) not in res:
        res.add((vlb1, e[2], vlb2))
    return res

  def getDFSEdge(self, e):
    return MyDFSEdge((e[0], e[1], (self.nodes[e[0]].label, e[2], self.nodes[e[1]].label)))
    
  def getMatchingNodeDict(self, label):  
    ''' func phi - return LIST of DICTIONARY {0: node_index} they have same label'''
    res = []
    for i in range(len(self.nodes)):
      if self.nodes[i].label == label:
        temp = {0: i}
        res.append(temp)
    return res

  def getMatchingNodeList(self, label):
    ''' return LIST of nodes's id whose label is 'label' '''
    res = []
    for i in range(len(self.nodes)):
      if self.nodes[i].label == label:
        res.append([i])
    return res
  
  def getLabelList(self):
    '''return LIST of label in the graph'''
    res = []
    for i in self.nodes:
      if i.label not in res:
        res.append(i.label)
    return res

  def getLabelNum(self):
    ''' return DICT {vlb: num of vlb} '''
    res = dict.fromkeys(self.getLabelList(), 0)
    for i in self.nodes:
      res[i.label] += 1
    return res

  def getLabelDict(self):
    ''' return  DICT {vlb: [node's ids]} '''
    res = dict.fromkeys(self.getLabelList(), None)
    for i in res:
      res[i] = []
    for i in self.nodes:
      res[i.label].append(i.id)
    return res


  def getLabelMatching(self):
    ''' return dictionary {vlb: [vlb to_nodes]}'''
    res = dict.fromkeys(self.getLabelList(), None)
    for i in res:
      res[i] = []
    for i in self.nodes:
      for j in i.to_nodes:
        if self.nodes[j].label not in res[i.label]:
          res[i.label].append(self.nodes[j].label)
    return res


  def getLabelMatchingFrom(self):
    ''' return dictionary {vlb: [vlb from_nodes]}'''
    res = dict.fromkeys(self.getLabelList(), None)
    for i in res:
      res[i] = []
    for i in self.nodes:
      for j in i.from_nodes:
        if self.nodes[j].label not in res[i.label]:
          res[i.label].append(self.nodes[j].label)
    return res
  
  def display(self):
    for n in self.nodes:
      print (n)
    for e in self.edges:
      print (e)
  
  def plot(self):
    #cycle = list(mcolors.CSS4_COLORS.keys())
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gnx = nx.DiGraph()
    vlbs = {}
    elbs = {}
    vids = {}
    for i in range(len(self.nodes)):
        gnx.add_node(i, label= self.nodes[i].label)
        vlbs[i] = self.nodes[i].label
        vids[i] = i
    for e in self.edges:
      gnx.add_edge(e[0], e[1], label=e[2])
      elbs[(e[0], e[1])] = e[2]
    
    vcolors = [cycle[i%len(cycle)] for i in vlbs.values()]
    ecolors = 'black'
    #ecolors = [cycle[len(cycle)%i] for i in elbs.values()]
    fsize = (min(12, 1 * len(self.nodes)),
              min(8, 1 * len(self.nodes)))
    plt.figure(3, figsize=fsize)
    pos = nx.circular_layout(gnx)
#     pos = nx.spring_layout(gnx)
    nx.draw_networkx(gnx, pos, node_size = 200, node_color = vcolors, edge_color = ecolors, labels = vids)
    #nx.draw_networkx_edge_labels(gnx, pos, edge_labels= elbs)
    plt.autoscale(enable = True)
    plt.show()

  def DFSUtil(self, v, visited, res): 
    visited[v]= True
    res.append(v)
    for i in self.nodes[v].to_nodes: 
        if visited[i] == False: 
            self.DFSUtil(i, visited, res) 

  def DFS(self): 
    res = []
    V = len(self.nodes) 
    visited =[False]*(V) 

    for i in range(V): 
        if visited[i] == False: 
            self.DFSUtil(i, visited, res) 
    return res

  def DFSUtilUndirected (self, v, visited, res):
    visited[v] = True
    neighbors = self.nodes[v].to_nodes.copy()
    neighbors.update(self.nodes[v].from_nodes)
    for i in neighbors:
      if visited[i] == False:
        res.append(v)
        self.DFSUtil(i, visited, res)

  def DFSUndirected (self, v):
    res = []
    visited = [False] * len (self.nodes)
    self.DFSUtilUndirected(v, visited, res)
    return res

  def DFSUtilvCode (self, v, visited, res):
    visited[v] = True
    for i in self.nodes[v].to_nodes:
      if visited[i] == False:
        res.append((v, i, VACANT_LABEL))
        self.DFSUtilvCode(i, visited, res)
    for i in self.nodes[v].from_nodes:
      if visited[i] == False:
        res.append((i, v, VACANT_LABEL))
        self.DFSUtilvCode(i, visited, res)

  def DFSvCode(self, v):
    res = []
    visited = [False] * len (self.nodes)
    self.DFSUtilvCode(v, visited, res)
    return res


  def toDFSMin(self):
    dfsMin = MyDFSCode()
    for i in range(self.getNumEdges()):
      temp, s = rightMostPatExt(dfsMin, self)
      if len(temp) == 0:
        return False
      dfsMin.append(s)
    return dfsMin

  def find_shortest_path(self, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    temp = self.nodes[start].to_nodes.copy()
    if len(temp) == 0 :
        return None
    shortest = None
    for node in temp:
        if node not in path:
            newpath = self.find_shortest_path(node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest

  def find_shortest_path_2(self, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    temp = self.nodes[start].to_nodes.copy()
    temp.update(self.nodes[start].from_nodes)
    if len(temp) == 0 :
        return None
    shortest = None
    for node in temp:
        if node not in path:
            newpath = self.find_shortest_path_2(node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest

  def drop_vertex_under_support(self, s):
    res = MyGraph()
    count_support_dict = self.getLabelNum() # count each label

    #drop vertex
    idx_map = {} #{idx_old: idx_new}  -  Memory: [V]*2*sizeof(int)[+epsilon]
    idx_vertex = 0
    for node in self.nodes: # O(|V|)
      if count_support_dict.get(node.label, -1) >= s:
        res.nodes.append(MyNode(idx_vertex, node.label))
        idx_map[node.id] =  idx_vertex
        idx_vertex += 1

    #drop edge
    for edge in self.edges: # O(|E|)
      if idx_map.get(edge[0], -1) != -1 and idx_map.get(edge[1], -1) != -1:
        res.addEdge(MyEdge((idx_map[edge[0]], idx_map[edge[1]], edge[2])))
        
    return res

  def subgraphIsomorphisms(self, c):   ## c: dfscode, g: graph
    res = []
    for i in range(len(self.nodes)):
      if self.nodes[i].label == c[0][2][0]:
        res.append([i])

    for i in c:         ## O(|c|)
      i1, i2, (vlb1, elb, vlb2) = i[0], i[1], i[2] 
      temp = []
      for j in res:     ## O(|V(g)|) 
        if i2 >= len(j):
          for xx in self.nodes[j[i1]].to_nodes:   ## O(|V(g|)
            if self.nodes[xx].label == vlb2 and xx not in j:
              j1 = j.copy()
              j1.append(xx)
              temp.append(j1.copy())
        elif i1 >= len(j):
          for xx in self.nodes[j[i2]].from_nodes:
            if self.nodes[xx].label == vlb1 and xx not in j:
              j1 = j.copy()
              j1.append(xx)
              temp.append(j1.copy())
        else:
          if MyEdge((j[i1], j[i2], elb)) in self.edges:
              temp.append(j.copy())
      res = temp.copy()
    return res

  ## O(|E(g)|.|V(g)|.|V(c)|.|c|) ~ O(|E(g)|.|V(g)|.|c|^2)
  def rightMostPatExt(self, dfscode):   
    rmNode = dfscode.getRMNode()  ## rightMost vertex of dfscode    ## O(|c|)
    res = set()   ## set of dfsEdges
    
    if rmNode < 0:   ## dfs is None
      for i in self.edges:    ## O(|E(g)|)
        res.add(MyDFSEdge((0, 1, (self.nodes[i[0]].label, i[2], self.nodes[i[1]].label))))
      temp = min(res)
      return res, temp
    else:
      minExt = [rmNode, rmNode, -1, -1]
      gc = dfscode.toGraph()    ## O(|c|^2)
      R = dfscode.getRMPathNodes() ## list of nodes in rightMostPath of dfscode  ## O(|c|^2)
      iso = self.subgraphIsomorphisms(dfscode)    ## O(|c|.|V(g)|^2)
      for i in iso:   ## i: dict          ## O(|E(g)|)
        ## backward ext
        ## backward edge
        for x in self.nodes[i[rmNode]].to_nodes:  ## x: node_index  ## O(|V(g)|)
          if x in i:             ## O(|V(c)|)
            v = i.index(x)
            if MyEdge((rmNode, v, VACANT_LABEL)) not in gc.edges and v in R:   ## O(|c|)
              res.add(MyDFSEdge((rmNode, v, (gc.nodes[rmNode].label, VACANT_LABEL, gc.nodes[v].label))))
              
              if minExt[0] > v:  #find min
                minExt[0] = v
        ## forward edge
        for x in self.nodes[i[rmNode]].from_nodes:
          if x in i:
            v = i.index(x)
            if MyEdge((v, rmNode, VACANT_LABEL)) not in gc.edges and v in R:   ## O(|c|)
              res.add(MyDFSEdge((v, rmNode, (gc.nodes[v].label, VACANT_LABEL, gc.nodes[rmNode].label))))
              
              if minExt[1] > v:
                minExt[1] = v
        ## forward ext
        for u in R:      ## O(|R|) ~ O(|V(c)|)
          ## forward edge
          for x in self.nodes[i[u]].to_nodes: ## O(|V(g)|)
            if x not in i:        ## O(|V(c)|) 
              res.add(MyDFSEdge((u, rmNode + 1, (gc.nodes[u].label, VACANT_LABEL, self.nodes[x].label))))
              
              if minExt[2] < u:   ## find max
                minExt[2] = u
          ## backward edge
          for x in self.nodes[i[u]].from_nodes:
            if x not in i:
              res.add(MyDFSEdge((rmNode + 1, u, (self.nodes[x].label, VACANT_LABEL, gc.nodes[u].label))))
              
              if minExt[3] < u:
                minExt[3] = u
    
    if min(minExt[0], minExt[1]) != rmNode:
      if minExt[0] <= minExt[1]:
        temp = MyDFSEdge((rmNode, minExt[0], (VACANT_INDEX, VACANT_LABEL, VACANT_INDEX)))
      else:
        temp = MyDFSEdge((minExt[1], rmNode,(VACANT_INDEX, VACANT_LABEL, VACANT_INDEX)))
    else:
      if minExt[2] >= minExt[3]:
        temp = MyDFSEdge((minExt[2], rmNode+1,(VACANT_INDEX, VACANT_LABEL, VACANT_INDEX)))
      else:
        temp = MyDFSEdge((rmNode + 1, minExt[3], (VACANT_INDEX, VACANT_LABEL, VACANT_INDEX)))
    
    for i in res:
      if temp[0] == i[0] and temp[1] == i[1]:
        if temp[2][0] == -1 or temp[2] > i[2]:
          temp = i

    return res, temp

  ## return list of dfscode generate from c
  def genCandidate(self, c): 
    res = set()
    R = c.getRMPathNodes()
    rmNode = c.getRMNode()
    gc = c.toGraph()
    lbMatchingTo = self.getLabelMatching()
    lbMatchingFrom = self.getLabelMatchingFrom()
    ## backward ext
    for i in R[1:]:
      if MyEdge((rmNode, i, VACANT_LABEL)) not in gc.edges:  ## be
        res.add(MyDFSEdge((rmNode, i, (gc.nodes[rmNode].label, VACANT_LABEL, gc.nodes[i].label))))
      if MyEdge((i, rmNode, VACANT_LABEL)) not in gc.edges:  ## fe
        res.add(MyDFSEdge((i, rmNode, (gc.nodes[i].label, VACANT_LABEL, gc.nodes[rmNode].label))))
    ## forward ext
    for u in R:
      for xx in lbMatchingTo[gc.nodes[u].label]:
        res.add(MyDFSEdge((u, rmNode + 1, (gc.nodes[u].label, VACANT_LABEL, xx))))
      for xx in lbMatchingFrom[gc.nodes[u].label]:
        res.add(MyDFSEdge((rmNode + 1,u, (xx, VACANT_LABEL, gc.nodes[u].label))))
    
    return res

  def MinerPattern2(self, s):
    QSet = dict()  ##  {vevlb: DFSCode resp}
    QMF = dict() ## {vevlb: [list of list: image of DFSCode]} all
    supps = dict()
    nodes = set()
    MF = dict () ## {vevlb: [list of list]} : frequent patterns
    for e in self.edges:
      dfse = self.getDFSEdge(e)
      if dfse[2] not in QMF:
        QMF[dfse[2]] = [[e[0], e[1]]]
      else:
        QMF[dfse[2]].append([e[0], e[1]])
    for key in QMF:
      sup = support2D(QMF[key])
      if sup >= s:
        QSet[key] = MyDFSCode([MyDFSEdge((0, 1, key))])
        supps[key] = sup
        nodes.update(v[0] for v in QMF[key])
        nodes.update(v[1] for v in QMF[key])
        MF[key] = QMF[key].copy()
    nodes = list(nodes)
    return QSet, supps, MF, nodes

  def pruneEdge(self, MF, nodes):
    newGraph = MyGraph()
    nodes_map = dict()
    for i in range(len(nodes)):
      nodes_map[nodes[i]] = i
      newGraph.nodes.append(MyNode(i, self.nodes[nodes[i]].label))
    for i in MF:
      index = 0
      for v in MF[i]:
        v0, v1 = nodes_map[v[0]], nodes_map[v[1]] 
        newGraph.addEdge(MyEdge((v0, v1, VACANT_LABEL)))
        MF[i][index] = [v0, v1].copy()
        index += 1 
    return newGraph

  def localMine(self, c, s, MS, index):   ## c: dfscode, g: graph
    node0 = MS[index].copy()
    count = 0
    res = []
    for i in range(c.getRMNode() + 1):
      res.append(set())

    for n0 in node0:
      aa = [-1] * len(res)
      aa[index] = n0
      iso = []
      iso.append(aa.copy())
      for i in c.extDFSCodeFromIndex(index):         ## O(|c|)
        i1, i2, (vlb1, elb, vlb2) = i[0], i[1], i[2] 
        temp = []
        for j in iso:     ## O(|V(g)|) 
          if j[i2] == -1:  ## fw ext
            for xx in self.nodes[j[i1]].to_nodes:   ## O(|V(g|)
              if self.nodes[xx].label == vlb2 and xx not in j:
                ##and getLabel(xx, u) == elb:## ignore label edge
                j[i2] = xx
                temp.append(j.copy())
          elif j[i1] == -1:  ## backward edge 
            for xx in self.nodes[j[i2]].from_nodes:
              # if ((i1 < len(MS) and xx in MS[i1]) or g.nodes[xx].label == vlb1) and xx not in j:
              if self.nodes[xx].label == vlb1 and xx not in j:
                j[i1] = xx
                temp.append(j.copy())
          else:
            if j[i2] in self.nodes[j[i1]].to_nodes:
              temp.append(j.copy())
        
        iso = temp.copy()
      
      if len(iso) == 0:
        count += 1
        if count > len(node0) - s:
          return -1, list(), -1
      else:
        for k in iso:
          for l in range(len(k)):
            res[l].add(k[l])
    
    for i in range(len(res)):
      if len(res[index]) > len(res[i]):
        index = i

    return len(res[index]), res, index


  def PatExt(self, c, s, temp, MS, index):
    cdds = self.genCandidate(c)
    for cd in cdds:
      new = c.mcopy()
      new.append(cd)
      if new.isMin():
        sup, newMS, newIndex = self.localMine(new, s, MS, index)
        if sup >= s:
          newtemp = MyCodeGraph(new, sup)
          temp.addChild(newtemp)
          self.PatExt(new, s, newtemp, newMS, newIndex)
        else:
          newMS.clear()

  def FPMiner(self, s):
    g1 = self.drop_vertex_under_support(s)
    QSet, s2, MF2, nodes = g1.MinerPattern2(s)
    g2 = g1.pruneEdge(MF2, nodes)
    res = MyCodeGraph()
    for i in QSet:
      temp = MyCodeGraph(QSet[i], s2[i])
      res.addChild(temp)
      MS = convertMFtoMS(MF2[i])
      index = 0 if len(MS[0]) <= len(MS[1]) else 1
      g2.PatExt(QSet[i], s, temp, MS, index)
    return res

  def genNewEdges(self, rule):
    res = set()
    labelDict = self.getLabelDict()

    ql, qr = rule[0], rule[1]
    iso = self.subgraphIsomorphisms(ql)
    for i in iso:
        isoo = []
        isoo.append(i.copy())
        for c in qr:
            i1, i2, (vlb1, elb, vlb2) = c[0], c[1], c[2] 
            temp = []
            for j in isoo:     ## O(|V(g)|) 
                if i2 >= len(j) and i1 >= len(j):
                    break
                if i2 >= len(j):
                  for xx in labelDict[vlb2]:   ## O(|V(g|)
                    if MyEdge((j[i1], xx, elb)) not in self.edges and xx not in j:
                        j1 = j.copy()
                        j1.append(xx)
                        temp.append(j1.copy())
                elif i1 >= len(j):
                  for xx in labelDict[vlb1]:
                    if MyEdge((xx, j[i2], elb)) not in self.edges and xx not in j:
                        j1 = j.copy()
                        j1.append(xx)
                        temp.append(j1.copy())
                else:
                  if MyEdge((j[i1], j[i2], elb)) not in self.edges:
                    temp.append(j.copy())
                    
            isoo = temp.copy()
        if (len(isoo) > 0):
            for k in isoo:
                for c in qr:
                    res.add(MyDFSEdge((k[c[0]], k[c[1]], c[2])))
    return res

  def plotNewEdges(self, newEdges):
    #cycle = list(mcolors.CSS4_COLORS.keys())
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gnx = nx.DiGraph()
    vlbs = {}
    vids = {}
    
    for i in range(len(self.nodes)):
      gnx.add_node(i, label= self.nodes[i].label)
      vlbs[i] = self.nodes[i].label
      vids[i] = i
    for e in self.edges:
        gnx.add_edge(e[0], e[1], color = 'black')
        
    for e in newEdges:
        gnx.add_edge(e[0], e[1], color = 'red')
    
    vcolors = [cycle[i%len(cycle)] for i in vlbs.values()]
    ecolors = [c for (u, v, c) in gnx.edges.data('color')]
    fsize = (min(12, 1 * len(self.nodes)),
              min(8, 1 * len(self.nodes)))
    plt.figure(3, figsize=fsize)
    pos = nx.circular_layout(gnx)
    # pos = nx.spring_layout(gnx)
    nx.draw_networkx(gnx, pos, node_size = 200, node_color = vcolors, edge_color = ecolors, labels = vids)
    #nx.draw_networkx_edge_labels(gnx, pos, edge_labels= elbs)
    plt.autoscale(enable = True)
    plt.show()


class MyCodeGraph():
  def __init__(self, data = None, supp = 0, children = None):
    if data is not None:
      assert isinstance(data, MyDFSCode)
      self.data = data.mcopy()
    else:
      self.data = data
    self.support = supp
    self.children = []
    if children is not None:
      for child in children:
        self.addChild(child)
  
  def addChild(self, child):
    assert isinstance(child, MyCodeGraph)
    self.children.append(child)
  
  def updateSupport(self, supp):
    self.support = supp

  def getLeafCodes(self):
    res = []
    def _get_leaf_nodes(node):
        if node is not None:
            if len(node.children) == 0:
                res.append((node.data.mcopy(), node.support))
            for n in node.children:
                _get_leaf_nodes(n)
    _get_leaf_nodes(self)
    return res
  
  def getAllCodes(self):
    res = []
    def _get_all_nodes(node):
        if node is not None:
          if node.data is not None:
            res.append(node)
          for n in node.children:
            _get_all_nodes(n)
    _get_all_nodes(self)
    return res

  def getAllData(self):
    res = []
    def _get_all_nodes(node):
        if node is not None:
          if node.data is not None:
            res.append((node.data.mcopy(), node.support))
          for n in node.children:
            _get_all_nodes(n)
    _get_all_nodes(self)
    return res

  def getAllDFSCode(self):
    res = []
    def _get_all_dfs(node):
        if node is not None:
          if node.data is not None:
            res.append(node.data.mcopy())
          for n in node.children:
            _get_all_dfs(n)
    _get_all_dfs(self)
    return res

  def getBFSearch(self):
    nodes = []
    queue = [self]
    while queue:
        cur_node = queue[0]
        queue = queue[1:]
        if cur_node.data is not None:
          nodes.append((cur_node.data.mcopy(), cur_node.support))
        for child in cur_node.children:
            queue.append(child)
    return nodes

  def getTrueLeaf(self):
    res = []
    leafs = self.getLeafCodes()
    for i in range(len(leafs)):
      flag = True
      for j in range(len(leafs)):
        if len(leafs[i][0]) < len(leafs[j][0]):
          iso = subgraphIsomorphisms(leafs[i][0], leafs[j][0].toGraph())
          if len(iso) != 0:
            flag = False
            break
      if flag:
        res.append(leafs[i])
    return res
    
  def getNum(self):
    if not isinstance(self, MyCodeGraph):
      return 0
    count = 1
    for i in self.children:
      count+= i.getNum()
    return count

  def __eq__(self, other):
    return (self.data == other.data and self.support == other.support and self.children == other.children)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    return '{} : {}'.format(self.data, self.support)
    
  def display (self):
    print (self)
    for i in self.children:
      i.display()


  def ruleGen(self, n):
    s, conf = [], []
    q = queue.Queue()
    gcList = self.getBFSearch()   ## get all (data, supp) in codeGraph by BFS
    for i in range(len(gcList)):
      q = queue.Queue()
      q.put(gcList[i])
      while not q.empty():
        qR = q.get()
        ancestors = genAncestors(gcList, qR[0])
        for ql, qr in ancestors:    ## get all ancestors ql of qR
          q.put(ql)
          if qR[1]/ql[1] >= n and qr.isConnected():
            if (ql[0], qr) not in s:
              s.append((ql[0], qr))
              conf.append(qR[1]/ql[1])
    return s, conf


class MyDFSCode(list):
  #def __init__(self):
    # self.rmPath = list() ##list index of vertex from rightmost vertex to root in DFSCode 

  def __eq__(self, other):
    la, lb = len(self), len(other)
    if la != lb:
        return False
    for i in range(la):
        if self[i] != other[i]:
            return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def getRMNode(self):
    res = -1
    for i in self:
      res = max(res, i[0], i[1])
    return res
  
  def toRMPath(self):
    ''' list of edge_index from right_most_vertex to root'''
    rmPath = []
    old = None
    for i in range(len(self)-1, -1, -1):
      if i == len(self) - 1:
        rmPath.append(i)
      temp = self[i]
      i1, i2 = temp[0], temp[1]
      if i1 < i2 and (old == None or old == i2):
        rmPath.append(i)
        old = i1
    return rmPath
    
  
  def getRMPathNodes(self):
    g = self.toGraph()
    end = self.getRMNode()
    res = g.find_shortest_path(0, end, [])
    if not res:
      res = g.find_shortest_path_2(0, end, [])
    res = list(set(res))
    return sorted(res, reverse = True)

  
  def toGraph(self):
    g = MyGraph()
    nodes = []
    nodes_map = dict()
    for i in self:   ## O(|c|)
      i1, i2, (v1, e, v2) = i[0], i[1], i[2]
      if i1 not in nodes:   ## O(|V(c)|)
        nodes.append(i1)
        nodes_map[i1] = len(nodes) - 1
        g.addNode(MyNode(nodes_map[i1], v1))
      if i2 not in nodes:
        nodes.append(i2)
        nodes_map[i2] = len(nodes) - 1
        g.addNode(MyNode(nodes_map[i2], v2))
      g.addEdge(MyEdge((nodes_map[i1], nodes_map[i2], e)))
    return g

  def isMin(self):
    g = self.toGraph()
    dfsMin = MyDFSCode()
    for i in range(len(self)):
      temp, s = g.rightMostPatExt(dfsMin)
      if len(temp) == 0:
        print ('isMin', dfsMin, self)
      if s != self[i]:
        return False
      dfsMin.append(s)
    return True

  
  def extDFSCodeFromIndex(self, index):
    res = []
    g = self.toGraph()
    extDFS = g.DFSvCode(index)
    for i in g.edges:
      if i not in extDFS:
        extDFS.append(i)
    for i in extDFS:
      res.append(g.getDFSEdge(i))
    return res

  def mcopy(self):
    res = type(self)()
    for i in self:
      res.append(i)
    return res

  def isConnected(self):
    g = self.toGraph()
    dfs = g.DFSUndirected(0)
    return len(dfs) == len(g.nodes) 

## Cacultating support from LIST 2D of MF
def support2D(img):
  if len(img) == 0:
    return 0
  temp = []
  for i in range(len(img[0])):
    temp.append(set())
  for i in range (len(img)):
    for j in range(len(img[i])):
      temp[j].add(img[i][j])
  res = len(temp[0])
  for i in temp:
    if res > len(i):
      res = len(i)
  return res

def convertMFtoMS(img):
  if len(img) == 0:
    return 0
  temp = []
  for i in range(len(img[0])):
    temp.append(set())
  for i in range (len(img)):
    for j in range(len(img[i])):
      temp[j].add(img[i][j])
  return temp

def genAncestors(gcList, dfscode):
  res = []
  for i in gcList:
    if len(i[0]) < len(dfscode):
      iso = dfscode.toGraph().subgraphIsomorphisms(i[0])
      if len(iso) > 0:
        nodes_map = iso[0].copy()
        dfs = MyDFSCode()
        for k in i[0]:
            dfse = MyDFSEdge((nodes_map[k[0]], nodes_map[k[1]], k[2]))
            dfs.append(dfse)
        qr = genConsequence(dfs, dfscode)
        
        newqr = MyDFSCode()
        for k in qr:
            if k[0] not in nodes_map:
              nodes_map.append(k[0])
            if k[1] not in nodes_map:
              nodes_map.append(k[1])
            newqr.append(MyDFSEdge((nodes_map.index(k[0]), nodes_map.index(k[1]), k[2])))
            
        res.append(((i[0].mcopy(), i[1]), newqr)) 
    else:
      break
  return res

def genConsequence(anc, code):
  res = MyDFSCode()
  for i in code:
    if i not in anc:
      res.append(i)
  return res

def toMinRule(ql, qr):
  g = MyGraph()
  nodes = []
  nodes_map = dict()
  for i in ql:   ## O(|c|)
    i1, i2, (v1, e, v2) = i[0], i[1], i[2]
    if i1 not in nodes_map:   ## O(|V(c)|)
      nodes.append(i1)
      nodes_map[i1] = len(nodes) - 1
      g.addNode(MyNode(nodes_map[i1], v1))
    if i2 not in nodes_map:
      nodes.append(i2)
      nodes_map[i2] = len(nodes) - 1
      g.addNode(MyNode(nodes_map[i2], v2))
    g.addEdge(MyEdge((nodes_map[i1], nodes_map[i2], e)))
  newql = g.toDFSMin()
  newqr = MyDFSCode()
  for i in qr:
    if i[0] not in nodes_map:
      nodes.append(i[0])
      nodes_map[i[0]] = len(nodes) - 1
    if i[1] not in nodes_map:
      nodes.append(i[1])
      nodes_map[i[1]] = len(nodes) - 1
    newqr.append(MyDFSEdge((nodes_map[i[0]], nodes_map[i[1]], i[2])))
  return newql, newqr
