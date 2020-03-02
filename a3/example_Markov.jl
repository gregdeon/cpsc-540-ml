# Load X and y variable
using JLD

# Load initial probabilities and transition probabilities of Markov chain
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])

# Set 'k' as number of states
k = length(p1)

# Confirm that initial probabilities sum up to 1
@show sum(p1)

# Confirm that transition probabilities sum up to 1, starting from each state
@show sum(pt,dims=2)

# Q1.1.1: generate samples from Markov chain
include("misc.jl")
function sampleAncestral(p1, pt, d)
	x = zeros(Int64, d)
	x[1] = sampleDiscrete(p1)
	for t in 2:d
		x[t] = sampleDiscrete(pt[x[t-1], :])
	end
	return x
end

# and use to generate Monte Carlo estimate of distribution at t = 50
mc_counts = zeros(k)
num_samples = 10000
for sample in 1:num_samples
	mc_counts[sampleAncestral(p1, pt, 50)[50]] += 1
end
mc_distribution = mc_counts / num_samples
@show mc_distribution

# Q1.1.2: compute exact marginals
function marginalCK(p1, pt, d)
	p = p1[:]
	for t in 2:d
		p = pt' * p
	end
	return p
end

# and find exact marginals at t = 50
ck_distribution = marginalCK(p1, pt, 50)
@show ck_distribution

# Q1.1.3: find most likely state at each time
# Compute up to t = 100 -- this makes the trend clear
mode_states = zeros(Int64, 100)
for t in 1:100
	mode_states[t] = argmax(marginalCK(p1, pt, t))
end
@show mode_states

# Q1.1.4: find optimal decoding
# TODO: implement Viterbi decoding

# Q1.1.5: find conditional probabilities with rejection sampling
num_samples = 10000
num_accepted = 0
mc_counts = zeros(k)

for sample in 1:num_samples
	x = sampleAncestral(p1, pt, 10)
	if x[10] == 6
		mc_counts[x[5]] += 1
		global num_accepted
		num_accepted += 1
	end
end

mc_distribution = mc_counts / num_accepted
@show mc_distribution
@show num_accepted
