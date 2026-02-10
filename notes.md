# November 18th - what have we learned?

uniform vs normal distribution for entity values don't seem to affect the final visualizations at all

pairwise_5 has little structure 

pairwise_10 has "stringy" structure for shallow depth but a more blob-shaped for 16

windowing depth = 8 -> 5000/2^7 approx 39 elements in deepest window layer 

seemingly the most structured results are for pairwise-10_depth-8 and any pairwise-all

# December
What happens if we take a neural network and we try to approximate the PCA curve? Or some regression
Can we learn the curve 
Get a model that is reusable -> keep the training
Estimate the manifold of the curve -> look into log regression/ linear regression, off the shelf neural network and learn the differences over the curve we learned -> we need the loss estimator for two inputs for a specific value 