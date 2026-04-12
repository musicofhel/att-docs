# Open Questions — reservoir-multimode

## Active
1. **Does OR(binding_direct, Crocker) push sr_drift above 0.984?**
   - Current best: 0.9837 (agent-2, binding only at STRIDE=400 with global PCA)
   - Expected: OR catches Crocker-missed early drift windows → sr_drift→1.0?
   - Testing in eval #7 (global PCA fix for binding)

## Resolved
1. ~~Why does STRIDE=200 hurt binding?~~ — autocorrelation inflates baseline sd (agent-1 Eval #6)
2. ~~Why does my noise regression happen?~~ — fixed seed for Crocker + STRIDE=200 (fixed in eval #6)
3. ~~Does inverted binding generalize to noise/cd?~~ — No. Binding only drops for sr_drift (Künneth)
4. ~~What is the F1 ceiling?~~ — ~0.99+ possible with STRIDE=400 + correct implementation

## Open Scientific Questions
1. **Why does connection_death F1=1.0 at STRIDE=400?**
   - With 20 drift windows and Crocker at STRIDE=400, catches all 20. 
   - cd topology changes RAPIDLY (connections die from step 5000) → big Crocker signal at 5200.

2. **Surrogate ablation surrogate_f1 per mode**
   - Expected ~0 (topology-specific). Needs to be confirmed from per_eval output.

3. **Extra seeds {3-7} validation**  
   - Can add to payload as extra_seed_f1 without affecting the 9 required pairs.
   - Not yet done by any agent.
