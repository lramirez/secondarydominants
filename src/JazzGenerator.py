#!/usr/bin/env python


from random import choice, random, shuffle, randint
import sys, getopt
import json
import pandas as pd
import numpy as np
import markovify
import torch
import ChordLanguageModel as CLM
from sdpychord import pychord as pc
import Chords2Midi
import MelodyModel

class JazzGenerator:
    def __init__(self, rnn_config_path = "lm_config.json", model_path = "../models/lm.pt", filename="jazz.mid", data_path="../data/chords3.csv", n_bars = 8, initstates=30, candidates=3000, n_tune_candidates = 20, chain_order=3, init_state_order = 2):
        self.RNN_CONFIG_PATH = rnn_config_path
        self.RANDOM_INIT_STATES = initstates
        self.N_CANDIDATES = candidates
        self.N_TUNE_CANDIDATES = n_tune_candidates
        self.CHAIN_ORDER = chain_order
        self.DATA_PATH = data_path
        self.N_BARS = 8
        self.OUTPUT_PATH = filename
        self.MODEL_PATH = model_path
        self.INIT_STATE_ORDER = init_state_order
        with open(self.RNN_CONFIG_PATH, 'r') as fp:
            self.rnn_config = json.load(fp)
        df = pd.read_csv(self.DATA_PATH)
        df.chords = df.chords.apply(lambda x: eval(x))
        self.mc_data = df.chords.to_numpy().tolist()
        self.mc = markovify.Chain(self.mc_data, state_size=self.CHAIN_ORDER)
        self.lm = CLM.ChordLanguageModel(self.DATA_PATH, self.rnn_config, model_path=self.MODEL_PATH)
    def generate_jazz(self):
        print("Generating some jazz...")
        material = self.mc_generate()

        # This is still being worked on - we're not actually looking at the LM outputs, just disregarding the top (and bottom) of the sorted list.
        # We need threshold values, and this needs human evaluation.
        # At least these ranges should be configurable, so the user could choose to hear basic II-V-I -stuff, or something a bit wilder or even disturbing.
        a_range = [0.03, 0.05]
        b_range = [0.05, 0.1]
        As = material[int(a_range[0]*len(material)):int(a_range[1]*len(material))] 
        Bs = material[int(b_range[0]*len(material)):int(b_range[1]*len(material))]
        tunes = []
        for i in range(self.N_TUNE_CANDIDATES):
            a = choice(As)[0]
            b = choice(Bs)[0]
            tune = a+a+b+a
            tunes.append(tune)
        # Evaluate tune candidates
        tunescores = self.lm.evaluate(tunes)
        best_idx = np.array(tunescores).argmax()
        tune = tunes[best_idx]
        mm = MelodyModel.MelodyModel()
        melody = [int(tone) for tone in mm.get_melody_for_chords(tune)]
        Chords2Midi.render(tune, self.OUTPUT_PATH, melody=melody)
        a = tune[:8]
        b = tune[16:24]
        autobop_a = " ".join([c+" %"+" |" for c in a])
        autobop_b = " ".join([c+" %"+" |" for c in b])
        autobop_sheet = "|| "+autobop_a+"\n| "+autobop_a+"|\n"
        autobop_sheet += "|| "+autobop_b+"|\n"
        autobop_sheet += "|| "+autobop_a+"|\n"
        return autobop_sheet
        
    def get_init_states(self, n, n_states):
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
                #print("added state", key)
                init_states.append(key)
        shuffle(init_states)
        return init_states[:n_states]
    def mc_generate(self):
        init_states = self.get_init_states(self.INIT_STATE_ORDER, self.RANDOM_INIT_STATES)
        outputs = []
        for state in init_states:
            walks = []
            for i in range(self.N_CANDIDATES):
                walk = self.mc.walk(init_state=state)
                if self.validate_walk_length(walk):
                    # TODO: CHECK FOR DUPLICATES
                    walks.append([w[1] for w in walk])
            if len(walks) > 0:
                outputs = outputs + walks
        scoredoutputs = []
        scores = self.lm.evaluate(outputs)
        scored = []
        for i in range(len(outputs)):
            scored.append((outputs[i], scores[i]))
        scored.sort(key=lambda x: x[1], reverse=True)
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
def main(argv):
    outfile = None
    try:
        opts, args = getopt.getopt(argv, "ho:", ["outfile="])
    except getopt.GetoptError:
        print("JazzGenerator -o outputfile")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("JazzGenerator -o <outputfile>")
            sys.exit()
        elif opt in ("-o", "--outfile"):
            outfile = arg
    if outfile:
        gen = JazzGenerator(filename=outfile)
    else:
        gen = JazzGenerator()
    jazz = gen.generate_jazz()
    print("Generated tune:")
    print(jazz)
    print("\nMidi saved to", gen.OUTPUT_PATH)

if __name__ == "__main__":
    main(sys.argv[1:])

 