% TODO: Should more POS types be allowed here?
__=subjpass [<2 /NN.*/ | <2 CD | <2 PRP | <2 VBG]
            < (__=passiveverb <1 acl
                              <2 VBN
                              !< ~subjpass
                              < (__=agent <1 nmod=agentdep
                                          < (/by_[0-9]+/=by <1 case=byrel <2 __=bypos)))

relabel agentdep nsubj
% Remove POS/edge label children of by before merging up any children
delete byrel
delete bypos
delete by
insert subjpass $- agent