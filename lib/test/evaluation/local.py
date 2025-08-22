from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/lasot'
    settings.network_path = '/data1/xuelin/xuelin/stark6/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/nfs'
    settings.otb_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/OTB2015'
    settings.prj_dir = '/data1/xuelin/xuelin/stark6'
    settings.result_plot_path = '/data1/xuelin/xuelin/stark6/test/result_plots'
    settings.results_path = '/data1/xuelin/xuelin/stark6/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data1/xuelin/xuelin/stark6'
    settings.segmentation_path = '/data1/xuelin/xuelin/stark6/test/segmentation_results'
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/trackingNet'
    settings.uav_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/UAV123'
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    
    settings.tnl2k_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/tnl2k'
    settings.otb_lang_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/otb_lang'
    settings.ref_coco_dir = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/ref_coco'
    settings.lasot_extension_subset_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/lasot_extension_subset'
    settings.uvot_path = '/data1/xuelin/xuelin/VideoX-master/SeqTrack/data/UVOT'

    return settings

