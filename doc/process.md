# In each time slot, handle state transition and reclaim resources

**The first two transitions shouldn't be handled in this process.**

- Determine which instance should failed in **previous time slot**, handle the transition: 

  - Normal→Backup;
  - Normal→Broken;
  - Backup→Broken.

  then reclaim the resource.

- Determine which sfc is expired, **if the state is broken, then don't need to bother it**, for it has been handled in previous process, handle the expired condition.

# When a stand-by instance need to be start

- Release the reserved resources occupied by this stand-by instance;

