# initial imports
from __future__ import division
from flask import Flask
from flask import render_template
from flask import request
from flask import flash
from flask import json

# local helper function imports
from subject import subject
from define_data import define_data
from session import session
import copy

# data imports
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.io import loadmat
import glob

# instance and config
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


#------------------------------------------------------------------------------

# database connection

# functions
def crp(recalls=None, subjects=None, listLength=None, lag_num=None, skip_first_n=0):
    """
    %CRP   Conditional response probability as a function of lag (lag-CRP).
    %
    %  lag_crps = crp(recalls_matrix, subjects, list_length, lag_num)
    %
    %  INPUTS:
    %  recalls_matrix:  a matrix whose elements are serial positions of recalled
    %                   items.  The rows of this matrix should represent recalls
    %                   made by a single subject on a single trial.
    %
    %        subjects:  a column vector which indexes the rows of recalls_matrix
    %                   with a subject number (or other identifier).  That is,
    %                   the recall trials of subject S should be located in
    %                   recalls_matrix(find(subjects==S), :)
    %
    %     list_length:  a scalar indicating the number of serial positions in the
    %                   presented lists.  serial positions are assumed to run
    %                   from 1:list_length.
    %
    %         lag_num:  a scalar indicating the max number of lag to keep track
    %
    %    skip_first_n:  an integer indicating the number of recall transitions to
    %                   to ignore from the start of the recall period, for the
    %                   purposes of calculating the CRP. this can be useful to avoid
    %                   biasing your results, as the first 2-3 transitions are
    %                   almost always temporally clustered. note that the first
    %                   n recalls will still count as already recalled words for
    %                   the purposes of determining which transitions are possible.
    %                   (DEFAULT=0)
    %
    %
    %  OUTPUTS:
    %        lag_crps:  a matrix of lag-CRP values.  Each row contains the values
    %                   for one subject.  It has as many columns as there are
    %                   possible transitions (i.e., the length of
    %                   (-list_length + 1) : (list_length - 1) ).
    %                   The center column, corresponding to the "transition of
    %                   length 0," is guaranteed to be filled with NaNs.
    %
    %                   For example, if list_length == 4, a row in lag_crps
    %                   has 7 columns, corresponding to the transitions from
    %                   -3 to +3:
    %                   lag-CRPs:     [ 0.1  0.2  0.3  NaN  0.3  0.1  0.0 ]
    %                   transitions:    -3   -2    -1   0    +1   +2   +3
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')
    if lag_num is None:
        lag_num = listLength - 1
    elif lag_num < 1 or lag_num >= listLength or not isinstance(lag_num, int):
        raise ValueError('Lag number needs to be a positive integer that is less than the list length.')
    if not isinstance(skip_first_n, int):
        raise ValueError('skip_first_n must be an integer.')

    # Convert recalls and subjects to numpy arrays
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects -- we will calculate a CRP for each
    usub = np.unique(subjects)
    # Number of possible lags = (listLength - 1) * 2 + 1; e.g. a length-24 list can have lags -23 through +23
    num_lags = 2 * listLength - 1
    # Initialize array to store the CRP for each subject (or other unique identifier)
    result = np.zeros((usub.size, num_lags))
    # Initialize arrays to store transition counts
    actual = np.empty(num_lags)
    poss = np.empty(num_lags)

    # For each subject/unique identifier
    for i, subj in enumerate(usub):
        # Reset counts for each participant
        actual.fill(0)
        poss.fill(0)
        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls[subjects == subj]))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(recalls[subjects == subj]):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k + 1] and k >= skip_first_n:
                    next_rec = trial_recs[k + 1]
                    pt = np.array([trans for trans in range(1 - rec, listLength + 1 - rec) if rec + trans not in seen], dtype=int)
                    poss[pt + listLength - 1] += 1
                    trans = next_rec - rec
                    # Record the actual transition that was made
                    actual[trans + listLength - 1] += 1

        result[i, :] = actual / poss
        result[i, poss == 0] = np.nan

    result[:, listLength - 1] = np.nan

    return result[:, listLength - lag_num - 1:listLength + lag_num]
def spc(recalls=None, subjects=None, listLength=None):
    """
    SPC   Serial position curve (recall probability by serial position).

    p_recall = spc(recalls, subjects, listLength)

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items. The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a 1D array which indexes the rows of the recalls matrix
                    with a subject number (or other identifier). That is,
                    the recall trials of subject S should be located in
                    recalls[subjects==S, :]

        listLength: a scalar indicating the number of serial positions in the
                    presented lists. serial positions are assumed to run
                    from [1, 2, ..., list_length].


    OUTPUTS:
        p_recall:  a matrix of probablities.  Its columns are indexed by
                   serial position and its rows are indexed by subject.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    # Convert inputs to numpy arrays if they are not already
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects
    usub = np.unique(subjects)
    # Create list of all serial positions (starting with 1)
    positions = np.arange(1, listLength+1)
    # We will return one SPC for each unique subject
    result = np.zeros((len(usub), listLength))

    for i, subj in enumerate(usub):
        # Select only the trials from the current subject
        subj_recalls = recalls[subjects == subj]
        # Create a "recalled" matrix of ones and zeroes indicating whether each presented item was correctly recalled
        subj_recalled = np.array([np.isin(positions, trial_data) for trial_data in subj_recalls])
        # Calculate the subject's SPC as the fraction of trials on which they recalled each serial position's item
        result[i, :] = subj_recalled.mean(axis=0)

    return result

def pfr(recalls, subjects, listLength):
    """"
    PFR   Probability of first recall.

    Computes probability of recall by serial position for the
    first output position.
    [p_recalls] = pfr(recalls_matrix, subjects, list_length)

    INPUTS:
        recalls:    a matrix whose elements are serial positions of recalled
                    items.  The rows of this matrix should represent recalls
                    made by a single subject on a single trial.

        subjects:   a column vector which indexes the rows of recalls_matrix
                    with a subject number (or other identifier).  That is,
                    the recall trials of subject S should be located in
                    recalls_matrix(find(subjects==S), :)

        listLength: a scalar indicating the number of serial positions in the
                    presented lists.  serial positions are assumed to run
                    from 1:list_length.

    OUTPUTS:
        p_recalls:  a matrix of probablities.  Its columns are indexed by
                    serial position and its rows are indexed by subject.
    """
    if recalls is None:
        raise Exception('You must pass a recalls matrix.')
    elif subjects is None:
        raise Exception('You must pass a subjects vector.')
    elif listLength is None:
        raise Exception('You must pass a list length.')
    elif len(recalls) != len(subjects):
        raise Exception('recalls matrix must have the same number of rows as subjects.')

    subject = np.unique(subjects)
    result = [[0] * listLength for count in range(len(subject))]
    for subject_index in range(len(subject)):
        count = 0
        for subj in range(len(subjects)):
            if subjects[subj] == subject[subject_index]:
                if recalls[subj][0] > 0 and recalls[subj][0] < 1 + listLength:
                    result[subject_index][recalls[subj][0] - 1] += 1
                count += 1
        for index in range(listLength):
            result[subject_index][index] /= count
    return result

def make_clean_recalls_mask2d(data):
    """makes a clean mask without repetition and intrusion"""
    result = copy.deepcopy(data)
    for num, item in enumerate(data):
        seen = []
        for index, recall in enumerate(item):

            if recall > 0 and recall not in seen:
                result[num][index] = 1
                seen.append(recall)
            else:
                result[num][index] = 0
    return result

def get_patients():
	files = glob.glob('./data/*.mat')
	patients = []
	for file in files:
		patients.append(file[7:file.find('_e')])

	return sorted(set(patients))

def load_session(path):
	x = loadmat(path)
	x = x['events'][0]

	return x

def get_data_info(patient):
	sessions = glob.glob('./data/' + patient + '_events_sess*.mat')
	sessions.sort()
	data_info = {
		'session_paths' : sessions,
	}

	return data_info

def getPRec(sessions):
	precs = []
	for sess in sessions:
		x = load_session(sess)
		recalled = len(x[x['recalled'] == 1])
		total = len(x[x['type'] == 'WORD'])
		precs.append(recalled/total)

	return precs

def getIntrusions(sessions):
	intrusions = []
	for sess in sessions:
		x = load_session(sess)
		intrution = len(x[x['intrusion'] == 1])
		total = len(x[x['type'] == 'WORD'])
		intrusions.append(intrution/total)

	return intrusions

def getRepeats(sessions):
	repeats = []
	for sess in sessions:
		x = load_session(sess)
		rec_words = x[x['type'] == 'REC_WORD']
		rec_words = rec_words['item'].tolist()
		total = len(x[x['type'] == 'WORD'])

		unique_recalls = set()
		for r in rec_words:
		    unique_recalls.add(r[0])
		if len(rec_words) > 0:
			repeat_rate = (len(rec_words) - len(unique_recalls)) / total
			repeats.append(repeat_rate)

	return repeats

def getAvgSPC(sessions):
	session_spc = []
	for sess in sessions:
		x = load_session(sess)
		# get recalled words
		rec_words = x[x['type'] == 'REC_WORD']
		# number of trials in this session
		num_trials = rec_words['trial'].max()[0][0]

		trial_spc = []
		for i in range(num_trials):
		    trial = rec_words[rec_words['trial'] == i+1]
		    serials = trial['serialPos']

		    # check for empty recall trial
		    if len(trial) == 0:
		    	continue

		    # flatten list of arrays to list
		    serials = np.concatenate(serials).ravel()
		    # remove nans
		    serials = np.array([ int(x) for x in serials if str(x) != 'nan'])
		    # remove negatives
		    serials = np.array([ int(x) for x in serials if x > 0])
		    # remove duplicates 
		    serials = list(set(serials))

		    serial_position_counts = np.bincount(serials, minlength=13) # don't hardcode
		    serial_position_counts = serial_position_counts[1:] # don't care about 0
		    trial_spc.append(np.array(serial_position_counts))

		session_spc.append(np.mean(np.array(trial_spc), axis=0))

	return np.mean(np.array(session_spc), axis=0).tolist()

def getAvgPFR(sessions):
	session_pfr = []
	for sess in sessions:
		x = load_session(sess)
		# get recalled words
		rec_words = x[x['type'] == 'REC_WORD']
		# number of trials in this session
		num_trials = rec_words['trial'].max()[0][0]

		first_recalls = []
		for i in range(num_trials):
		    trial = rec_words[rec_words['trial'] == i+1]
		    serials = trial['serialPos']
		    
		    # check for empty recall trial
		    if len(trial) == 0:
		        continue
		    
		    # flatten list of arrays to list
		    serials = np.concatenate(serials).ravel()
		    
		    # remove nans
		    serials = np.array([ int(x) for x in serials if str(x) != 'nan'])
		    # remove negatives
		    serials = np.array([ int(x) for x in serials if x > 0])

			# creates a serial position array for each trial and marks which position was first recalled		    	
		    first_recall = np.zeros(12)

		    try:
		    	first_recall[serials[0]-1] += 1
		    except IndexError:
		    	continue

		    first_recalls.append(first_recall)

		session_pfr.append(np.mean(np.array(first_recalls), axis=0))

	return np.mean(np.array(session_pfr), axis=0).tolist()

def getLagCRP(sessions):
	session_spc = []
	lags = []
	for sess in sessions:
		x = load_session(sess)
		# get recalled words
		rec_words = x[x['type'] == 'REC_WORD']
		# number of trials in this session
		num_trials = rec_words['trial'].max()[0][0]

		for i in range(num_trials):
		    trial = rec_words[rec_words['trial'] == i+1]
		    serials = trial['serialPos']

		    # check for empty recall trial
		    if len(trial) == 0:
		    	continue

		    # flatten list of arrays to list
		    serials = np.concatenate(serials).ravel()
		    # remove nans
		    serials = np.array([ int(x) for x in serials if str(x) != 'nan'])
		    # remove negatives
		    serials = np.array([ int(x) for x in serials if x > 0])
		    # remove duplicates
		    serials = list(set(serials))
		    if len(serials) > 0:
		    	lags.append( crp([serials], ['single patient'], 12)[0] )

	lags = np.nanmean(lags, 0)
	lags[np.isnan(lags)] = 0
	indices = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	lag_crp = { str(indices[n]):round(lag, 3) for n,lag in enumerate(lags) }
	return lag_crp

# routes
@app.route("/")
def main():
	patients = get_patients()
	return render_template('index.html',
							patients=patients)

@app.route('/patient/id/<string:user_id>')
def profile(user_id):
    patient_data_paths = get_data_info(user_id)
    num_sessions = len(patient_data_paths['session_paths'])

    if num_sessions:
    	session_precs = getPRec(patient_data_paths['session_paths'])
    	session_intrusions = getIntrusions(patient_data_paths['session_paths'])
    	session_repeats = getRepeats(patient_data_paths['session_paths'])
    	average_spc = getAvgSPC(patient_data_paths['session_paths'])
    	average_pfr = getAvgPFR(patient_data_paths['session_paths'])
    	lag_crp = getLagCRP(patient_data_paths['session_paths'])

    	# round array values before sending -- might want to move this to the functions
    	session_precs = [ round(x, 3) for x in session_precs ]
    	session_intrusions = [ round(x, 3) for x in session_intrusions ]
    	session_repeats = [ round(x, 3) for x in session_repeats ]
    	average_spc = [ round(x, 3) for x in average_spc ]
    	average_pfr = [ round(x, 3) for x in average_pfr ]

    	return render_template('data.html', 
    							patient_id=user_id, 
    							num_sessions=num_sessions, 
    							sessions=patient_data_paths['session_paths'], 
    							precs=session_precs,
    							intrusions=session_intrusions,
    							repeats=session_repeats,
    							avg_spc=json.dumps(average_spc),
    							avg_pfr=json.dumps(average_pfr),
    							lag_crp=json.dumps(lag_crp),
	    					  )
    else:
    	return render_template('error.html')

@app.route("/examplePost", methods=['POST'])
def examplePost():
	user_id = request.values.get("user_id")
	patient_data_paths = get_data_info(user_id)
    # creates a subject object from ID, which has many cool methods
	try:
		su = subject(index=0, subjectList=[user_id]).concatenate(False, patient_data_paths['session_paths'])
		scs = round(np.mean(su.SCS), 3)
		tcs = round(np.mean(su.TCS), 3)
	except ValueError as err:
		print(err)
		scs = '<i>Error: ' + str(err) + '</i>'
		tcs = '<i>Error: ' + str(err) + '</i>'
	
	data = {
		"scs":str(scs),
		"tcs":str(tcs)
	}
	return json.dumps(data)


#------------------------------------------------------------------------------


if __name__ == "__main__":
	app.run()


















