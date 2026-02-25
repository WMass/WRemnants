#!/bin/bash
export APPTAINER_BIND="/scratch,/cvmfs,/etc/pki/tls/certs,/etc/grid-security/certificates" 
if [[ -d $WREM_BASE ]]; then
    export APPTAINER_BIND="${APPTAINER_BIND},${WREM_BASE}/.."
fi
CONTAINER=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:v46

# Kerberos cache setup
# Assuming kinit was already done on the host!
KRB5CC_HOST_DIR="/run/user/$UID/krb5ccdir"
KRB5CC_CONTAINER_DIR="/tmp/krb5ccdir"

# Ensure kerberos permissions for eos access (requires systemd kerberos setup)
if [[ -d "$KRB5CC_HOST_DIR" ]]; then
    export APPTAINER_BIND="${APPTAINER_BIND},${KRB5CC_HOST_DIR}:${KRB5CC_CONTAINER_DIR}"
    export KRB5CCNAME="DIR:${KRB5CC_CONTAINER_DIR}"
else
    echo "⚠️ Warning: Kerberos cache directory $KRB5CC_HOST_DIR does not exist!"
fi

singularity run $CONTAINER "$@"
