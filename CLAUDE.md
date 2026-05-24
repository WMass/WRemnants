# Environment Setup

Commands in this repository must be run inside the apptainer/singularity container at:

```
/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling
```

or the corresponding docker/podman container at

```
gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling
```

Check whether you are already inside the container by checking if the `SINGULARITY_CONTAINER` or $container environment variable is set.

If not inside the container, prefix commands with `scripts/ci/run_with_singularity.sh`:

```bash
scripts/ci/run_with_singularity.sh bash -c "source setup.sh && <command>"
```

If already inside the container, then commands can be run directly (but still sourcing setup.sh).

## Verification

To verify the environment is correctly set up, run:

```bash
python scripts/tests/testenv.py
```

Expected output ends with:
```
Successfully imported relevant packages
```
