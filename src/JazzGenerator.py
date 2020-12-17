from random import choice, random, shuffle, randint
import sys
import json
import pandas as pd
import numpy as np
import markovify
import torch
import ChordLanguageModel as CLM
from sdpychord import pychord as pc
import Chords2Midi

class JazzGenerator:
    def __init__(self, filename="jazz.mid", data_path="../data/chords3.csv", n_bars = 8, initstates=20, candidates=500, pr=-18, pr_range=1, chain_order=3):
        self.RNN_CONFIG_PATH = "lm_config.json"
        self.RANDOM_INIT_STATES = initstates
        self.N_CANDIDATES = candidates
        self.pr = pr
        self.pr_range = pr_range
        self.CHAIN_ORDER = chain_order
        self.DATA_PATH = data_path
        self.N_BARS = 8
        self.OUTPUT_PATH = filename
        with open(self.RNN_CONFIG_PATH, 'r') as fp:
            self.rnn_config = json.load(fp)
        df = pd.read_csv(self.DATA_PATH)
        df.chords = df.chords.apply(lambda x: eval(x))
        self.mc_data = df.chords.to_numpy().tolist()
        self.mc = markovify.Chain(self.mc_data, state_size=self.CHAIN_ORDER)
        self.lm = CLM.ChordLanguageModel(self.DATA_PATH, self.rnn_config, model_path="../models/lm.pt")
    def generate_jazz(self, song=False):
        print("Generating MC walks...")
        material = self.mc_generate()
        print("...done.")
        a_range = 0.1
        b_range = [0.1, 0.2]

        # We need a better way to choose these. Perhaps human evaluation + simple regression model
        # would help us make better choices?
        As = material[:int(a_range*len(material))]
        Bs = material[int(b_range[0]*len(material)):int(b_range[1]*len(material))]
        a = choice(As)[0]
        b = choice(Bs)[0]
        tune = a+a+b+a
        print("Rendering MIDI-file...")
        Chords2Midi.render(tune, self.OUTPUT_PATH)
        print("...done.")
        autobop_a = " ".join([c+" "+c+" |" for c in a])
        autobop_b = " ".join([c+" "+c+" |" for c in b])
        autobop_sheet = "|| "+autobop_a+"\n|"+autobop_a+"|\n"
        autobop_sheet += "|| "+autobop_b+"|\n"
        autobop_sheet += "|| "+autobop_a+"|\n"
        return autobop_sheet
        
    def get_init_states(self, n):
        print("Retrieving initial states...")
        init_states = []
        for key in self.mc.model.keys():
            tail = key[:self.CHAIN_ORDER-n]
            chords = key[self.CHAIN_ORDER-n:]
            fail = False
            for t in tail:
                if not t == '___BEGIN__':
                    fail = True
                    break
            for c in chords:
                if c == '___BEGIN__':
                    fail = True
                    break
            if not fail:
                init_states.append(key)
        shuffle(init_states)
        print("...done.")
        return init_states[:n]
    def mc_generate(self):
        init_states = self.get_init_states(self.RANDOM_INIT_STATES)
        outputs = []
        print("Generating chains...")
        for state in init_states:
            walks = []
            for i in range(self.N_CANDIDATES):
                walk = self.mc.walk(init_state=state)
                if self.validate_walk_length(walk):
                    walks.append([w[1] for w in walk])
            if len(walks) > 0:
                outputs = outputs + walks
        print("...done.")
        scoredoutputs = []
        print("Getting scores...")
        scores = self.lm.evaluate(outputs)
        scored = []
        for i in range(len(outputs)):
            scored.append((outputs[i], scores[i]))
        scored.sort(key=lambda x: x[1], reverse=True)
        print("...done.")
        return scored
    def validate_time(self, c):
        bars = 0
        time = 0
        fail = False
        for chord in c:
            time += 1/int(chord[0])
            if time == 1:
                bars = bars + 1
                time = 0
                continue
            if time > 1:
                fail = True
                break
        if not fail:
            if bars == self.N_BARS:
                return True
        else:
            return False
    def validate_walk_length(self, c):
        if len(c) == self.N_BARS:
            return True
    