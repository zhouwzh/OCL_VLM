python preprocess.py
mv /home/wz3008/slot-attn/output /home/wz3008/slot-attn/sqf_root_movi_a_pair/output
mksquashfs sqf_root_movi_a_pair movi_a_pair.sqf -noappend -comp zstd -Xcompression-level 15 -b 1M