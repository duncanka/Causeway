__=subjpass < nsubjpass=subjpassdep
            > (__ <2 /VB*/ < (/be_[0-9]+/=be <1 auxpass=berel <2 __=bepos)
                           < (/by_[0-9]+/=by <1 __=byrel <2 __=bypos < (__=agent <1 pobj=agentdep)))

relabel agentdep nsubj
relabel subjpassdep dobj
move agent $+ subjpass
% Remove POS/edge label children of by and be before merging up any children
delete berel
delete bepos
delete byrel
delete bypos
% Delete by and be and integrate their children. Nowhere better to put them than into the parent, if they exist. (Probably parse errors anyway.)
excise by by
excise be be