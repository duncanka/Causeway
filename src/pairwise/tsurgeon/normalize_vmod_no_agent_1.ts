__=subjpass [<2 /NN.*/ | <2 /CD*/ | <2 PRP | <2 VBG]
            < (__=passiveverb <1 vmod
                              <2 VBN
                              !< (/by_[0-9]+/ < (__ <1 pobj))
                              !< (__ <1 dobj)
                              !< vmod_marker)

% stupid marker to avoid infinite loops
insert (vmod_marker) >-1 passiveverb
insert subjpass >3 passiveverb