import os 
import numpy as np 
from utils import * 

import plotly.subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


subjects = {}
labels = {}

for file in os.listdir(SEGMENT_DIR):
	data = np.load(os.path.join(SEGMENT_DIR,file),allow_pickle=True).item()
	# print(data)
	
	# print(data)

	 
	# scores = list(data['score'][data['score'] != np.inf])
	if not np.isinf(data['score']):
		scores = [data['score']]
	else: 
		continue

	subject,label,iteration = file.split('.')[0].split('_')
	
	# print(file,subject,label,iteration,scores)

	if subject not in subjects: 
		subjects[subject] = []
	subjects[subject].extend(scores)
	
	if label not in labels:
		labels[label] = []
	
	labels[label].extend(scores)



# fig = go.Figure()

# for subject in subjects:
#     fig.add_trace(go.Violin(x=[subject]*len(subjects[subject]),
#                             y=subjects[subject],
#                             name=subject,
#                             box_visible=True,
#                             meanline_visible=True))




fig = ff.create_distplot([subjects[subject] for subject in subjects], list(subjects.keys()))

fig.show()

fig = ff.create_distplot([labels[label] for label in labels], list(labels.keys()))

fig.show()

# fig = go.Figure()

# for label in labels:
#     fig.add_trace(go.Violin(x=[label]*len(labels[label]),
#                             y=labels[label],
#                             name=label,
#                             box_visible=True,
#                             meanline_visible=True))

# fig.show()




print(subjects)
print(labels)