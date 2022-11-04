#! /bin/sh

#basedir='/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/'
#basedir='/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05918_2022-10-20/'
#basedir='/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p1_ga/'
#basedir='/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06018_2022-11-02'
#basedir='/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06024_2022-11-03'
basedir='/data/data_mrcv2/99_GSR/SMS_Testing/LIFEVOLUNTEER_06041_2022-11-04'
cd ${basedir}

# for dir in 06024*; do
#    echo ""
#    echo "${dir}"
#    echo ""
#    cd ${dir}
#    #sms_2dpc_recon ScanArchive*.h5
#    cd SMS_2DPC
#    sms_recon_simple.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 1 --smap_type lowres --sms_phase 1 --out_filename InPhase.h5
#    sms_recon_simple.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 1 --smap_type lowres --sms_phase -1 --out_filename OutPhase.h5
#    cd ${basedir}
# done
for dir in *pwv-radial_SMS; do
    echo ""
    echo "${dir}"
    echo ""
    cd ${dir}
    #sms_2dpc_recon ScanArchive*.h5
    cd SMS_2DPC
    sms_recon_simple.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 20 --resp_gate --smap_type lowres --sms_phase 1 --out_filename InPhase.h5
    sms_recon_simple.py --filename MRI_Raw.h5 --recon_type pils --gate_type ecg --frames 20 --resp_gate --smap_type lowres --sms_phase -1 --out_filename OutPhase.h5
    cd ${basedir}
done

# cd /export/home/groberts/CODE/PRECON/flow_recon/