q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))

#applies rms norm to the last dimension
#scores = (Q @ K.T) / sqrt(dim)
#           ↑ grows      ↑ fixed constant
#insight on this improvement is that as Q and K increase the scoring
#calculation becomes unstable.
#sqrt(dim) stays consistent but Q and K increases
#to keep all variables within an appropriate range, we normalize Q and K

# scores = (Q @ K.T) / sqrt(dim)
# as Q and K grow, scores become extreme
# softmax on extreme values → near one-hot → gradients vanish
# rms_norm fixes Q and K magnitude to 1
# so sqrt(dim) remains a reliable reference point throughout training