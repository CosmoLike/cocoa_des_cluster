### unlink emulator likelihoods from the usual cocoa likelihood
if [ -d "$ROOTDIR/projects/des_cluster/emulator_deprecated/likelihood" ]; then
	rm $ROOTDIR/projects/des_cluster/likelihood/*_emu.py
    rm $ROOTDIR/projects/des_cluster/likelihood/*_emu.yaml
fi


unset SPDLOG_LEVEL
