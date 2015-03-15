% TODO: Should more POS types be allowed here?
__=subjpass [<2 /NN.*/ | <2 CD | <2 PRP | <2 VBG]
            < (__=passiveverb <1 vmod
                              <2 VBN
                              < (/by_[0-9]+/=by < (__=agent <1 pobj=agentdep)))

relabel agentdep nsubj
move agent >3 passiveverb
insert subjpass >4 passiveverb
delete by