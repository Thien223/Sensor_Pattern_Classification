import math
import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  ### disable warning messages




def load_data(path):
	'''
	load data from path, return dataframe and index of each pattern
	:param path: path to file
	:return: (dataframe, index list)
	'''
	data = pd.read_csv(path)
	in_index = data.index[(data[' comment'] == 'IN')].tolist()
	out_index = data.index[(data[' comment'] == 'OUT')].tolist()
	_index = sorted(in_index+out_index)

##### split data to patterns
	# data_list = []
	# start_index=0
	# for idx in _index:
	# 	temp = data.iloc[start_index:idx]
	# 	data_list.append(temp)
	# 	start_index = idx
	return data, _index #, in_index, out_index


def padding(series, length=100, position='last', pad_value=None):
	new_series = []
	if position=='last':
		for serie in series:
			print(serie)
			origin_length = serie.shape[0]
			padding_length = length - origin_length
			padding_value = pad_value if pad_value is not None else serie[-1]
			padding_serie = np.full((padding_length,),padding_value)
			## create new series by concatenating serie and padding serie
			### append new serie to new_series list
			new_series.append(np.concatenate((serie, padding_serie), axis=0))
		return new_series
	else:
		for serie in series:
			origin_length = serie.shape[0]
			padding_length = length - origin_length
			padding_value = pad_value if pad_value is not None else serie[-1]
			padding_serie = np.full((padding_length,),padding_value)
			## create new series by concatenating padding serie and serie (append to first possition)
			### append new serie to new_series list
			new_series.append(np.concatenate((padding_serie,serie), axis=0))
		return new_series



###https://github.com/alexminnaar/time-series-classification-and-clustering/blob/master/Time%20Series%20Classification%20and%20Clustering.ipynb


def dynamic_time_wraping_distance(serie_1, serie_2, window=10):
	### calculate distance between 2 series
	DTW = {}
	w = max(window, abs(len(serie_1) - len(serie_2)))

	for i in range(-1, len(serie_1)):
		for j in range(-1, len(serie_2)):
			DTW[(i, j)] = float("inf")
	DTW[(-1, -1)] = 0

	for i in range(len(serie_1)):
		for j in range(max(0, i - w), min(len(serie_2), i + w)):
			dist = (serie_1[i] - serie_2[j]) ** 2
			DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
	return math.sqrt(DTW[len(serie_1) - 1, len(serie_2) - 1])


def lb_keogh_distance(s1, s2, r):
	### calculate distance between 2 series, the faster method
	"""
	use to check lower bound and upper bound between 2 time series before calculating dynamic time wrapper.
	---> save DTW calculating time
	:param s1:
	:param s2:
	:param r:
	:return:
	"""
	LB_sum = 0
	for ind, i in enumerate(s1):
		lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
		upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
		if i > upper_bound:
			LB_sum = LB_sum + (i - upper_bound) ** 2
		elif i < lower_bound:
			LB_sum = LB_sum + (i - lower_bound) ** 2
	return math.sqrt(LB_sum)

def k_means_clust(series, centroids=None, num_clusters=3, num_iter=100, w=10):
	'''
	clustering series into centroids groups. Because of average centroid calculating, this requires series has same length.
	:param series: input series
	:param centroids: list of pre defined centroids (for assign series to one of them), default is None, mean centroid will be newly calculated
	:param num_clusters: number of cluster to group (k)
	:param num_iter: training steps
	:param w:
	:return:
	'''
	### cluster a set of series to [num_clusters] cluster
	import random
	centroids = random.sample(list(series), num_clusters) if centroids is None else centroids
	assignments = None
	for n in range(num_iter):
		print("Training centroids - step {} / {}".format(n, num_iter))
		assignments = {}
		# assign data points to clusters
		for idx, serie in enumerate(series):
			min_distance = float("inf")  ### create a infinity number and assign to min
			closest_centroid_idx = None
			for centroid_idx, centroid_serie in enumerate(centroids):
				if lb_keogh_distance(serie, centroid_serie, w) < min_distance:
					current_serie_distance_from_centroid = dynamic_time_wraping_distance(serie, centroid_serie, w)
					if current_serie_distance_from_centroid < min_distance:
						min_distance = current_serie_distance_from_centroid
						closest_centroid_idx = centroid_idx
			#### add to assignments
			if closest_centroid_idx is not None:  ##### exclude empty series
				assignments.setdefault(closest_centroid_idx, [])
				assignments[closest_centroid_idx].append(idx)
			else:
				assignments.setdefault(closest_centroid_idx, [])
		# recalculate centroids of clusters
		for key in assignments:
			clust_sum = 0
			for k in assignments[key]:
				clust_sum = clust_sum + series[k]
			centroids[key] = [m / len(assignments[key]) for m in clust_sum]
	print("Training done..")
	return centroids, assignments


def frequency_transform(series, freq_number):
	series = data_list_
	import librosa


def train():
	paths=['dataset/1_person.csv','dataset/2_people.csv','dataset/3_people.csv']
	data_list = []
	labels = []
	for path in paths:
		data, _index = load_data(path)
		label = path.split('/')[-1].split('.')[0]
		#### split data to patterns

		start=0

		for end in _index:
			temp = data.iloc[start:end]
			start = end
			# start_index = idx
			# temp.plot()
			temp['Sum'] = temp[' durationA'] + temp[' durationB']
			temp['Diff'] = np.abs(temp[' durationA'] - temp[' durationB'])
			temp['Avg'] = (temp[' durationA'] + temp[' durationB']) / 2
			temp['Square'] = np.square(temp[' durationA'] + temp[' durationB'])
			temp['Sqrt'] = np.sqrt(temp[' durationA'] + temp[' durationB'])
			temp['log_Avg'] = np.log((temp[' durationA'] + temp[' durationB']) / 2)
			temp['Moving_Avg'] = temp['Avg'].ewm(span=10, adjust=False).mean()
			temp = temp.reset_index()

			data_list.append(np.asarray(temp['Moving_Avg'].dropna()))
			labels.append(label)
			# temp['Sum'].plot(grid=True, legend=True, label='Sum', title='1 person')
			# temp['Diff'].plot()
			# temp['Moving_Avg'].plot(grid=True, legend=True, label='Moving_Avg{}'.format(start), title='3 person')
			# temp['Square'].plot()
			# temp['Sqrt'].plot()


	length=len(max(data_list, key=len))
	### padding for consistent length
	data_list_ = padding(data_list, length=length, position='last',pad_value=220)
	# for serie in data_list_:
	# 	plt.plot(serie)
	centroids, assignments = k_means_clust(data_list_)
	#
	# ### PLOT RESULTS
	# #### save figure
	# ### save ploting images (we remove add building name to figure for clearly viewing)
	# print("Saving images..")
	# # Turn interactive plotting off
	# import matplotlib
	# matplotlib.use("TkAgg")  ##
	# import matplotlib.pyplot as plt
	# plt.ion()  ## turn plotting off
	# os.makedirs("images", exist_ok=True)
	# for i in range(3):
	# 	## images saving path
	# 	image_path = os.path.join("images", "cluster_{}.png".format(i))
	# 	## create image
	# 	plt.figure()
	# 	## plot centroid serie as a black line
	# 	## plot series that belong to current cluster
	# 	try:
	# 		for id in assignments[i]:
	# 			plt.plot(list(data_list_[id]), label=labels[id])
	# 	except:
	# 		pass
	# 	plt.plot(centroids[i], label="cluster-{}".format(i), color="black")
	# 	plt.savefig(image_path, format="png")
	# 	plt.legend()
	# 	plt.close()


if __name__=='__main__':
	train()