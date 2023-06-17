library(DiagrammeR)

grViz("digraph flowchart {
      graph [layout = neato]
      # node definitions with substituted label text
      node [fontname = TimesNewRoman, shape = rectangle, font=12]        
      tab1 [label = '@@1']
      tab2 [label = '@@2']
      tab3 [label = '@@3']
      tab4 [label = '@@4']
      tab5 [label = '@@5']
      tab6 [label = '@@6']
      tab7 [label = '@@7']
      tab8 [label = '@@8']
      tab9 [label = '@@9']
      tab10 [label = '@@10']
      tab11 [label = '@@11']
      

      # edge definitions with the node IDs
      tab1 -> tab2 -> tab3 -> tab4 -> tab5-> tab6 -> tab7 -> tab8 -> tab9 -> tab10 -> tab11;
      }

      [1]: 'Tweets collection'
      [2]: 'Data preprocessing'
      [3]: 'Grid search: Optimal number of topics'
      [4]: 'Topic Extraction with LDA'
      [5]: 'Data segmentation into topics'
      [6]: 'Rename Topics with Keywords'
      [7]: 'Compute Topic Overlap'
      [8]: 'Compute confidence matrix'
      [9]: 'Build FCM with Topics as nodes'
      [10]: 'Define initial activation vectors'
      [11]: 'Run simulations'

      
      
      ")
