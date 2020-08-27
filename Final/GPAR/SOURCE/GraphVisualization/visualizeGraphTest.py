# 3D visualize - graph data
import Thesis_Undirected_GPARs_Final as undirected
import Thesis_Directed_GPARs_Final as directed

import igraph as ig
import chart_studio.plotly as py
import plotly.graph_objs as go

import sys, getopt
import argparse

def GraphVisualizer(profiles_filename, relationships_filename, directed_=False):
  if directed_:
    g = directed.MyGraph()
  else:
    g = undirected.MyGraph()

  g.loadData(profiles_filename, relationships_filename)
  print('NumberOfNodes: ', len(g.nodes), 'NumberOfEdges: ', len(g.edges))

  Edges = []
  for edge in g.edges:
    i1, i2, lb = edge
    Edges.append((i1, i2))

  labels=[]
  idxs=[]
  N = len(g.nodes)
  print('len(N) = ', N)
  for node in g.nodes:
      labels.append(node.label)
      idxs.append('id: '+str(node.id)+' - label: '+str(node.label))
  if directed_:
    G = ig.Graph(Edges, directed=True)
  else:
    G = ig.Graph(Edges, directed=False)
  layt = G.layout('drl', dim=3) # random_3d, drl, kk_3d
  print('layt', layt)

  if profiles_filename == '../DATASETS/GenTest2/genTest2_profiles.txt':
    Xn=[layt[k][0] for k in range(3998)]
    Yn=[layt[k][1] for k in range(3998)]
    Zn=[layt[k][2] for k in range(3998)]
    Xe=[]
    Ye=[]
    Ze=[]
  else:
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    Zn=[layt[k][2] for k in range(N)]
    Xe=[]
    Ye=[]
    Ze=[]
  for e in Edges:
      Xe+=[layt[e[0]][0],layt[e[1]][0], None]
      Ye+=[layt[e[0]][1],layt[e[1]][1], None]
      Ze+=[layt[e[0]][2],layt[e[1]][2], None]

  trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=1),
                hoverinfo='none'
                )

  trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                              size=6,
                              color=labels,
                              colorscale='Viridis',
                              line=dict(color='rgb(50,50,50)', width=0.5)
                              ),
                text=idxs,
                hoverinfo='text'
                )

  axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

  dataName = profiles_filename.split('/')[-1]
  dataName = dataName.split('_')
  layout = go.Layout(
          title=dataName[0]+" - 3D visualization",
          width=1400,
          height=750,
          showlegend=False,
          scene=dict(
              xaxis=dict(axis),
              yaxis=dict(axis),
              zaxis=dict(axis),
          ),
      margin=dict(
          t=100
      ), )

  data=[trace1, trace2]
  fig=go.Figure(data=data, layout=layout)

  print('plot')
  fig.show()

def main(argv):
   relationships_filename = None
   profiles_filename = None
   directed = None

   '''
   try:
      opts, args = getopt.getopt(argv,"h:p:r:d",["ProfilesFileName","RelationshipFileName=","Directed"])
   except getopt.GetoptError:
      print ('*.py -p <profiles file_path> -r <relationships file_path> -d <Directed>')
      sys.exit(2)

   for opt, arg in opts:
      print(opt, arg)
      if opt in ("-h", "--help"):
         print ('*.py -p <profile file_path> -r <relationships file_path> -d <directed>')
         sys.exit()
      elif opt in ("-p", "--profiles"):
         profiles_filename = arg
      elif opt in ("-r", "--relationships"):
         relationships_filename = arg
      elif opt in ("-d", "--directed"):
         directed = arg'''

   parser = argparse.ArgumentParser()
   parser.add_argument('-p', '--profiles')
   parser.add_argument('-r', '--relationships')
   parser.add_argument('-d', dest='directed', action='store_true')
   args = parser.parse_args()

   '''
   print(directed)
   if directed == 'True':
      print('DIRECTED')
      directed = True
   else:
      print('UNDIRECTED')
      directed = False'''

   profiles_filename = args.profiles
   relationships_filename = args.relationships
   directed = args.directed

   assert (profiles_filename!=None and relationships_filename!=None)

   print(profiles_filename, relationships_filename, directed)  
   GraphVisualizer(profiles_filename, relationships_filename, directed)
   

if __name__=='__main__':
  main(sys.argv[1:])
