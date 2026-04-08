# Changelog

All notable changes to this project will be documented here.

### [2026-03-24]
- initial repo setup and basic cnn model structure
- added federated averaging loop but its super slow right now

### [2026-03-26]
- fixed bug where client weights werent averaging correctly (math error in aggregate function lol)
- added basic plotting for accuracy 

### [2026-03-28]
- added non-IID split after reading the FedAvg paper (data heterogeneity is crazy important for edge)
- tried learning rate 1e-3, too unstable, back to 2e-5 for now
- changed optimizer from adam to sgd locally

### [2026-04-01]
- added random client dropout simulation
- cleaned up file structure for submission
