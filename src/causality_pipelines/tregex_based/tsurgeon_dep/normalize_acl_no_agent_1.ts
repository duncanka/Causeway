__=subjpass [<2 /NN.*/ | <2 /CD*/ | <2 PRP | <2 VBG]
            < (__=passiveverb !<- acl_marker
                              <1 acl
                              <2 VBN
				              !< (/by_[0-9]+/ < (__ <1 pobj))
                              !< (__ <1 dobj))

% Insert marker to prevent newly inserted copy of tree from re-triggering pattern
insert (acl_marker) >-1 passiveverb
insert subjpass >3 passiveverb