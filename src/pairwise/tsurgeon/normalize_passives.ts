__=subjpass < nsubjpass=subjpassdep
            > (__ <2 /VB*/ < (/be_[0-9]+/=be <1 auxpass)
                           < (/by_[0-9]+/=by < (__=agent <1 pobj=agentdep)))

relabel agentdep nsubj
relabel subjpassdep dobj
move agent $+ subjpass
delete by
delete be