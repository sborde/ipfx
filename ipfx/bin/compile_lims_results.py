import numpy as np
import pandas as pd
import argschema as ags
import glob
import os.path
import ipfx.lims_queries as lq
import allensdk.core.json_utilities as ju
import logging

class CompileOutputParameters(ags.ArgSchema):
    ids = ags.fields.List( ags.fields.Integer,
        description="List of specimen IDs to process",
        cli_as_single_argument=True,
        default=None, allow_none=True
    )
    input_file = ags.fields.InputFile(
        description=("Input file of specimen IDs (one per line)"
                     "- optional if LIMS is source"),
        default=None, allow_none=True
    )
    project = ags.fields.String(
        description="Project code used for LIMS query, e.g. hIVSCC-MET",
        default="hIVSCC-MET",
        allow_none=True
    )
    cell_count_limit = ags.fields.Integer(
        description="Limit to number of cells evaluated",
        default=1000
    )
    output_file = ags.fields.OutputFile(
        description="output file path",
        default="compiled_output_ipfx_lims.csv",
        allow_none=True
    )

# from cell_features.long_squares
ls_features = [
    "input_resistance",
    "tau",
    "v_baseline",
    "sag",
    "rheobase_i",
    "fi_fit_slope",
]
hero_sweep_features = [
    'adapt',
    'avg_rate',
    'latency',
    'first_isi',
    'mean_isi',
    "median_isi",
    "isi_cv",
]
rheo_sweep_features = [
    'latency',
    'first_isi',
    'avg_rate',
]
mean_sweep_features = [
    'adapt',
]
spike_features = [
    'upstroke_downstroke_ratio',
    'threshold_v',
    'peak_v',
    'fast_trough_v',
    'trough_v',
    # include all troughs?
    # 'slow_trough_v',
    # TODO: add features not already averaged
    # 'width',
    # 'upstroke',
    # 'downstroke',
]
ramp_spike_features = [
    'threshold_i',
]
ls_spike_features = [
    # not in cell record
    'width',
    'upstroke',
    'downstroke',
]
invert = ["first_isi"]
spike_threshold_shift = ["trough_v", "fast_trough_v", "peak_v"]


def extract_local_pipeline_output(output_json):
    output = ju.read(output_json)
    record = {}
    cell_state = output.get('qc', {}).get('cell_state')
    if cell_state is not None:
        record['failed_qc'] = cell_state.get('failed_qc', False)
        record['fail_tags'] = '; '.join(cell_state.get('fail_tags'))

    fx_dict = output.get('feature_extraction')
    if fx_dict is not None:
        record.update(extract_fx_output(fx_dict))
    return record

def extract_fx_output(fx_dict, v2=False):
    record = {}
    cell_state = fx_dict.get('cell_state')
    if cell_state is not None:
        # record['failed_fx'] = cell_state.get('failed_fx', False)
        record['fail_fx_message'] = cell_state.get('fail_fx_message')

    if v2:
        cell_features = fx_dict["specimens"][0].get('cell_ephys_features', {})
    else:
        cell_features = fx_dict.get('cell_features', {})

    if cell_features["ramps"]:
        mean_spike_0 = cell_features["ramps"]["mean_spike_0"]
        add_features_to_record(spike_features + ramp_spike_features, mean_spike_0, record, suffix="_ramp")

    if cell_features["short_squares"]:
        mean_spike_0 = cell_features["short_squares"]["mean_spike_0"]
        add_features_to_record(spike_features, mean_spike_0, record, suffix="_short_square")

    long_squares = cell_features.get('long_squares')
    if long_squares is not None:
        add_features_to_record(ls_features, long_squares, record)

        sweep = long_squares.get('rheobase_sweep',{})
        add_features_to_record(rheo_sweep_features, sweep, record, suffix='_rheo')
        add_features_to_record(spike_features + ls_spike_features, sweep["spikes"][0], record, suffix="_rheo")

        sweep = long_squares.get('hero_sweep',{})
        add_features_to_record(hero_sweep_features, sweep, record, suffix='_hero')
        add_features_to_record(spike_features + ls_spike_features, sweep["spikes"][0], record, suffix="_hero")

        sweeps = long_squares.get('spiking_sweeps',{})
        for feature in mean_sweep_features:
            key = feature+'_mean'
            feat_list = [sweep[feature] for sweep in sweeps if feature in sweep]
            record[key] = np.mean([x for x in feat_list if x is not None])

    offset_feature_values(spike_threshold_shift, record, "threshold_v")
    # invert_feature_values(invert, record)
    return record

def offset_feature_values(features, record, relative_to):
    for feature in features:
        matches = [x for x in record if x.startswith(feature)]
        for match in matches:
            suffix = match[len(feature):]
            val = record.pop(match)
            record[match+"_rel"] = (val - record[relative_to+suffix]) if val is not None else None

def invert_feature_values(features, record):
    for feature in features:
        matches = [x for x in record if x.startswith(feature)]
        for match in matches:
            suffix = match[len(feature):]
            val = record.pop(match)
            record[match+"_inv"] = 1/val if val is not None else None

def add_features_to_record(features, feature_data, record, suffix=""):
    record.update({feature+suffix: feature_data.get(feature) for feature in features})

def compile_lims_results(specimen_ids):
    records = []
    for cell in specimen_ids:
        path = get_fx_output_json(cell)
        if path.startswith('/'):
            record = extract_fx_output(ju.read(path), v2=("V2" in path) or ("DATAFIX" in path))
            record["specimen_id"] = cell
            records.append(record)
    ephys_df = pd.DataFrame.from_records(records, index="specimen_id")
    return ephys_df

def get_specimen_ids(ids=None, input_file=None, project="T301", include_failed_cells=False, cell_count_limit=float('inf')):
    if ids is not None:
        specimen_ids = ids
    elif input_file is not None: 
        with open(module.args["input_file"], "r") as f:
            ids = [int(line.strip("\n")) for line in f]
        module.args.pop('ids')
    else:
        specimen_ids = lq.project_specimen_ids(
            project, passed_only=not include_failed_cells)
    if len(specimen_ids) > cell_count_limit:
        specimen_ids = specimen_ids[:cell_count_limit]
    logging.info(
        "Number of specimens to process: {:d}".format(len(specimen_ids)))
    return specimen_ids

def get_fx_output_json(specimen_id):
    """
    Find in LIMS the full path to the json output of the feature extraction module
    If more than one file exists, then chose the latest version

    Parameters
    ----------
    specimen_id

    Returns
    -------
    file_path: string
    """
    NO_SPECIMEN = "No_specimen_in_LIMS"
    NO_OUTPUT_FILE = "No_feature_extraction_output"
    
    sql = """
    select err.storage_directory, err.id
    from specimens sp
    join ephys_roi_results err on err.id = sp.ephys_roi_result_id
    where sp.id = %d
    """ % specimen_id

    res = lq.query(sql)
    if res:
        err_dir = res[0]["storage_directory"]

        file_list = glob.glob(os.path.join(err_dir, '*EPHYS_FEATURE_EXTRACTION_*_output.json'))
        if file_list:
            latest_file = max(file_list, key=os.path.getctime)   # get the most recent file
            return latest_file
        else:
            return NO_OUTPUT_FILE
    else:
        return NO_SPECIMEN

def main(ids=None, input_file=None, project="T301", include_failed_cells=False, cell_count_limit=float('inf'), output_file=None, **kwargs):
    specimen_ids = get_specimen_ids(ids, input_file, project, include_failed_cells, cell_count_limit)
    compile_lims_results(specimen_ids).to_csv(output_file)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CompileOutputParameters)
    main(**module.args)