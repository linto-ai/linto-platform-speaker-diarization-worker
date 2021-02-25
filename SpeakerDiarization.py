#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Speaker Diarization
from pyBK.diarizationFunctions import *
import librosa
import time
import webrtcvad
import logging
import librosa
import os
import sys
import uuid
import yaml
import numpy as np
from flask_swagger_ui import get_swaggerui_blueprint
##############


class SpeakerDiarization:
    def __init__(self):
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

        self.log = logging.getLogger(
            '__speaker-diarization-worker__' + '.' + __name__)

       # MFCC FEATURES PARAMETERS
        self.frame_length_s = 0.025
        self.frame_shift_s = 0.01
        self.num_bins = 30
        self.num_ceps = 30
        #####

        # Segment
        self.seg_length = 100  # Window size in frames
        self.seg_increment = 100  # Window increment after and before window in frames
        self.seg_rate = 100  # Window shifting in frames
        #####

        # KBM
        # Minimum number of Gaussians in the initial pool
        self.minimumNumberOfInitialGaussians = 1024
        self.maximumKBMWindowRate = 50  # Maximum window rate for Gaussian computation
        self.windowLength = 200  # Window length for computing Gaussians
        self.kbmSize = 320  # Number of final Gaussian components in the KBM
        # If set to 1, the KBM size is set as a proportion, given by "relKBMsize", of the pool size
        self.useRelativeKBMsize = 1
        # Relative KBM size if "useRelativeKBMsize = 1" (value between 0 and 1).
        self.relKBMsize = 0.3
        ######

        # BINARY_KEY
        self.topGaussiansPerFrame = 5  # Number of top selected components per frame
        self.bitsPerSegmentFactor = 0.2  # Percentage of bits set to 1 in the binary keys
        ######

        # CLUSTERING
        self.N_init = 16  # Number of initial clusters
        # Set to one to perform linkage clustering instead of clustering/reassignment
        self.linkage = 0
        # Linkage criterion used if linkage==1 ('average', 'single', 'complete')
        self.linkageCriterion = 'average'
        # Similarity metric: 'cosine' for cumulative vectors, and 'jaccard' for binary keys
        self.metric = 'cosine'
        ######

        # CLUSTERING_SELECTION
        # Distance metric used in the selection of the output clustering solution ('jaccard','cosine')
        self.metric_clusteringSelection = 'cosine'
        # Method employed for number of clusters selection. Can be either 'elbow' for an elbow criterion based on within-class sum of squares (WCSS) or 'spectral' for spectral clustering
        self.bestClusteringCriterion = 'elbow'
        self.sigma = 1  # Spectral clustering parameters, employed if bestClusteringCriterion == spectral
        self.percentile = 40
        self.maxNrSpeakers = 10  # If known, max nr of speakers in a sesssion in the database. This is to limit the effect of changes in very small meaningless eigenvalues values generating huge eigengaps
        ######

        # RESEGMENTATION
        self.resegmentation = 1  # Set to 1 to perform re-segmentation
        self.modelSize = 6  # Number of GMM components
        self.nbIter = 10  # Number of expectation-maximization (EM) iterations
        self.smoothWin = 100  # Size of the likelihood smoothing window in nb of frames
        ######

        # SWAGGER PARAM
        self.SWAGGER_URL = '/api-doc'
        self.SWAGGER_PATH = '/opt/swagger/swagger.yml'

    def swaggerUI(self, app):
        ### swagger specific ###
        if os.path.exists(self.SWAGGER_PATH):
            swagger_yml = yaml.load(
                open(self.SWAGGER_PATH, 'r'), Loader=yaml.Loader)
            swaggerui = get_swaggerui_blueprint(
                # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
                self.SWAGGER_URL,
                self.SWAGGER_PATH,
                config={  # Swagger UI config overrides
                    'app_name': "STT API Documentation",
                    'spec': swagger_yml
                }
            )
            app.register_blueprint(swaggerui, url_prefix=self.SWAGGER_URL)
            return True
        else:
            return False
        ### end swagger specific ###

    def compute_feat_Librosa(self, audioFile):
        try:
            filename = str(uuid.uuid4())
            self.file_path = "/tmp/"+filename
            audioFile.save(self.file_path)

            self.data, self.sr = librosa.load(self.file_path, sr=None)
            os.remove(self.file_path)

            frame_length_inSample = self.frame_length_s * self.sr
            hop = int(self.frame_shift_s * self.sr)
            NFFT = int(2**np.ceil(np.log2(frame_length_inSample)))
            if self.sr >= 16000:
                mfccNumpy = librosa.feature.mfcc(y=self.data,
                                                 sr=self.sr,
                                                 dct_type=2,
                                                 n_mfcc=self.num_ceps,
                                                 n_mels=self.num_bins,
                                                 n_fft=NFFT,
                                                 hop_length=hop,
                                                 fmin=20,
                                                 fmax=7600).T
            else:
                mfccNumpy = librosa.feature.mfcc(y=self.data,
                                                 sr=self.sr,
                                                 dct_type=2,
                                                 n_mfcc=self.num_ceps,
                                                 n_mels=self.num_bins,
                                                 n_fft=NFFT,
                                                 hop_length=hop).T

        except Exception as e:
            self.log.error(e)
            raise ValueError(
                "Speaker diarization failed when extracting features!!!")
        else:
            return mfccNumpy

    def computeVAD_WEBRTC(self, data, sr, nFeatures):
        try:
            if sr not in [8000, 16000, 32000, 48000]:
                data = librosa.resample(data, sr, 16000)
                sr = 16000

            va_framed = py_webrtcvad(
                data, fs=sr, fs_vad=sr, hoplength=30, vad_mode=0)
            segments = get_py_webrtcvad_segments(va_framed, sr)
            maskSAD = np.zeros([1, nFeatures])
            for seg in segments:
                start = int(np.round(seg[0]/self.frame_shift_s))
                end = int(np.round(seg[1]/self.frame_shift_s))
                maskSAD[0][start:end] = 1
        except Exception as e:
            self.log.error(e)
            raise ValueError(
                "Speaker diarization failed while voice activity detection!!!")
        else:
            return maskSAD

    def getSegments(self, frameshift, finalSegmentTable, finalClusteringTable, dur):
        numberOfSpeechFeatures = finalSegmentTable[-1, 2].astype(int)+1
        solutionVector = np.zeros([1, numberOfSpeechFeatures])
        for i in np.arange(np.size(finalSegmentTable, 0)):
            solutionVector[0, np.arange(
                finalSegmentTable[i, 1], finalSegmentTable[i, 2]+1).astype(int)] = finalClusteringTable[i]
        seg = np.empty([0, 3])
        solutionDiff = np.diff(solutionVector)[0]
        first = 0
        for i in np.arange(0, np.size(solutionDiff, 0)):
            if solutionDiff[i]:
                last = i+1
                seg1 = (first)*frameshift
                seg2 = (last-first)*frameshift
                seg3 = solutionVector[0, last-1]
                if seg.shape[0] != 0 and seg3 == seg[-1][2]:
                    seg[-1][1] += seg2
                elif seg3 and seg2 > 0.3:  # and seg2 > 0.1
                    seg = np.vstack((seg, [seg1, seg2, seg3]))
                first = i+1
        last = np.size(solutionVector, 1)
        seg1 = (first-1)*frameshift
        seg2 = (last-first+1)*frameshift
        seg3 = solutionVector[0, last-1]
        if seg3 == seg[-1][2]:
            seg[-1][1] += seg2
        elif seg3 and seg2 > 0.3:  # and seg2 > 0.1
            seg = np.vstack((seg, [seg1, seg2, seg3]))
        seg = np.vstack((seg, [dur, -1, -1]))
        seg[0][0] = 0.0
        return seg

    def format_response(self, segments):
        #########################
        # Response format is
        #
        # {
        #   "speakers":[
        #       {
        #           "id":"spk1",
        #           "tot_dur":10.5,
        #           "nbr_segs":4
        #       },
        #       {
        #           "id":"spk2",
        #           "tot_dur":6.1,
        #           "nbr_segs":2
        #       }
        #   ],
        #   "segments":[
        #       {
        #           "seg_id":1,
        #           "spk_id":"spk1",
        #           "seg_begin":0,
        #           "seg_end":3.3,
        #       },
        #       {
        #           "seg_id":2,
        #           "spk_id":"spk2",
        #           "seg_begin":3.6,
        #           "seg_end":6.2,
        #       },
        #   ]
        # }
        #########################

        json = {}
        _segments = []
        _speakers = {}
        seg_id = 1

        # Remove the last line of the segments.
        # It indicates the end of the file and segments.
        if segments[len(segments)-1][2] == -1:
            segments=segments[:len(segments)-1]

        for seg in segments:
            segment = {}
            segment['seg_id'] = seg_id
            segment['spk_id'] = 'spk'+str(int(seg[2]))
            segment['seg_begin'] = float("{:.2f}".format(seg[0])) 
            segment['seg_end'] = float("{:.2f}".format(seg[0] + seg[1]))  
            
            if segment['spk_id'] not in _speakers:
                _speakers[segment['spk_id']] = {}
                _speakers[segment['spk_id']]['spk_id'] = segment['spk_id']
                _speakers[segment['spk_id']]['duration'] = float("{:.2f}".format(seg[1])) 
                _speakers[segment['spk_id']]['nbr_seg'] = 1
            else:
                _speakers[segment['spk_id']]['duration'] += seg[1]
                _speakers[segment['spk_id']]['nbr_seg'] += 1
                _speakers[segment['spk_id']]['duration'] = float("{:.2f}".format(_speakers[segment['spk_id']]['duration'])) 

            _segments.append(segment)
            seg_id += 1

        json['speakers'] = list(_speakers.values())
        json['segments'] = _segments
        return json

    def run(self, audioFile):
        try:
            old_stdout = sys.stdout # backup current stdout
            sys.stdout = open(os.devnull, "w")
        
            start_time = time.time()
            feats = self.compute_feat_Librosa(audioFile)
            nFeatures = feats.shape[0]
            duration = nFeatures * self.frame_shift_s

            if duration < 5:
                return [[0, duration, 1],
                        [duration, -1, -1]]

            maskSAD = self.computeVAD_WEBRTC(self.data, self.sr, nFeatures)
            maskUEM = np.ones([1, nFeatures])

            mask = np.logical_and(maskUEM, maskSAD)
            mask = mask[0][0:nFeatures]
            nSpeechFeatures = np.sum(mask)
            speechMapping = np.zeros(nFeatures)
            # you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
            # so that we don't lose features on the way
            speechMapping[np.nonzero(mask)] = np.arange(1, nSpeechFeatures+1)
            data = feats[np.where(mask == 1)]
            del feats

            segmentTable = getSegmentTable(
                mask, speechMapping, self.seg_length, self.seg_increment, self.seg_rate)
            numberOfSegments = np.size(segmentTable, 0)
            # create the KBM
            # set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
            if np.floor((nSpeechFeatures-self.windowLength)/self.minimumNumberOfInitialGaussians) < self.maximumKBMWindowRate:
                windowRate = int(np.floor(
                    (np.size(data, 0)-self.windowLength)/self.minimumNumberOfInitialGaussians))
            else:
                windowRate = int(self.maximumKBMWindowRate)

            if windowRate == 0:
                #self.log.info('The audio is to short in order to perform the speaker diarization!!!')
                return [[0, duration, 1],
                        [duration, -1, -1]]

            poolSize = np.floor((nSpeechFeatures-self.windowLength)/windowRate)
            if self.useRelativeKBMsize:
                kbmSize = int(np.floor(poolSize*self.relKBMsize))
            else:
                kbmSize = int(self.kbmSize)

            # Training pool of',int(poolSize),'gaussians with a rate of',int(windowRate),'frames'
            kbm, gmPool = trainKBM(
                data, self.windowLength, windowRate, kbmSize)

            #'Selected',kbmSize,'gaussians from the pool'
            Vg = getVgMatrix(data, gmPool, kbm, self.topGaussiansPerFrame)

            #'Computing binary keys for all segments... '
            segmentBKTable, segmentCVTable = getSegmentBKs(
                segmentTable, kbmSize, Vg, self.bitsPerSegmentFactor, speechMapping)

            #'Performing initial clustering... '
            initialClustering = np.digitize(np.arange(numberOfSegments), np.arange(
                0, numberOfSegments, numberOfSegments/self.N_init))

            #'Performing agglomerative clustering... '
            if self.linkage:
                finalClusteringTable, k = performClusteringLinkage(
                    segmentBKTable, segmentCVTable, self.N_init, self.linkageCriterion, self.metric)
            else:
                finalClusteringTable, k = performClustering(
                    speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, self.bitsPerSegmentFactor, kbmSize, self.N_init, initialClustering, self.metric)

            #'Selecting best clustering...'
            if self.bestClusteringCriterion == 'elbow':
                bestClusteringID = getBestClustering(
                    self.metric_clusteringSelection, segmentBKTable, segmentCVTable, finalClusteringTable, k, self.maxNrSpeakers)
            elif self.bestClusteringCriterion == 'spectral':
                bestClusteringID = getSpectralClustering(self.metric_clusteringSelection, finalClusteringTable,
                                                         self.N_init, segmentBKTable, segmentCVTable, k, self.sigma, self.percentile, self.maxNrSpeakers)+1

            if self.resegmentation and np.size(np.unique(finalClusteringTable[:, bestClusteringID.astype(int)-1]), 0) > 1:
                finalClusteringTableResegmentation, finalSegmentTable = performResegmentation(data, speechMapping, mask, finalClusteringTable[:, bestClusteringID.astype(
                    int)-1], segmentTable, self.modelSize, self.nbIter, self.smoothWin, nSpeechFeatures)
                seg = self.getSegments(self.frame_shift_s, finalSegmentTable, np.squeeze(
                    finalClusteringTableResegmentation), duration)
            else:
                return [[0, duration, 1],
                        [duration, -1, -1]]

            sys.stdout = old_stdout # reset old stdout

            self.log.info("Speaker Diarization took %d[s] with a speed %0.2f[xRT]" %
                          (int(time.time() - start_time), float(int(time.time() - start_time)/duration)))
        except ValueError as v:
            self.log.error(v)
            raise ValueError('Speaker diarization failed during processing the speech signal')
        except Exception as e:
            self.log.error(e)
            raise Exception('Speaker diarization failed during processing the speech signal')
        else:
            return seg
