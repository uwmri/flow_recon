#! /bin/sh

basedir='/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2019_LIFE_PWV/phantom/Aorta_PWVPhantom_06509_2023-02-08/HR60_v3/'
cd ${basedir}

for dir in *pwv-radial_SMS; do
    echo ""
    echo "${dir}"
    echo ""

    cd ${dir}
    #sms_2dpc_recon ScanArchive*.h5

    cd SMS_2DPC
    llr_recon_flow.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 40 --smap_type espirit --sms_phase 1  --flow_processing --out_filename AAo.h5
    llr_recon_flow.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 40 --smap_type espirit --sms_phase -1 --flow_processing --out_filename AbdAo.h5
    cd ${basedir}
done