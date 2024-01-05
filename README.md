# CAR-Transformer

## Data

### WikiLingua

Download [here](https://drive.google.com/file/d/1PM7GFCy2gJL1WHqQz1dzqIDIEN6kfRoi/view?usp=sharing). Then put the downloaded `WikiLingua_data_splits.tar` into `data` folder and unpack it.

### GlobalVoice

1. Download [here](https://ucd1b0302a869f9bfea274a9f120.dl.dropboxusercontent.com/cd/0/get/CAhMu8RLd06SZ6iZQdQWjh96Is635UVITI2gJ7MQsGS6WPpUUS-GxLwVQkP8UlHU_-YZ1aQwLPt4KaVCMPCv2_hcbjaGxBTMG92MEL8LyugAOPzsBx9eFtP8TrCuA-ACEHIyXJ7ES0CxczeFkoswNwTKiKyXhyHNcKlojrSvWy8QwGYGdoEUnM91VgWZh2rvbh8/file?_download_id=1351457283938650712407520762227864344896363590492906009486732732938&_notify_domain=www.dropbox.com&dl=1) and put the downloaded `globalvoices_v0.zip` into `data` and unpack it.
2. Preprocess by running `python data_preprocess/proc_gv.py`.
3. Generate parallel monolingual summaries by running `bash ./data_preprocess/generate_parallel_gv.sh`. (You may need to modify the `batch_size`, `num_gpus`, and `num_workers` in `data_preprocess/generate_parallel.py` before running)

### CrossSum

1. Download [here](https://doc-00-c4-docs.googleusercontent.com/docs/securesc/nascbs2fm16f6dvlum0n441fc2nntt28/0i42l7atkprh9n1hv4urhrl0a59vmgr2/1690278600000/13979054203160936572/02343706292923644565/11yCJxK5necOyZBxcJ6jncdCFgNxrsl4m?e=download&ax=AGtFMPXpXLG-k7jgPxx9UXFyJsG87JlWLWdJbzdAHZRz_G2FeX-in9Z0Vt2z6_055meps_i-I5PLWv1ChMHpMGIphuT4WhoTNkq3ngo2hJYyfjvdQ0F8dfOsntCGDZ7RBD4RLr9sCIiU416P5ScsXnONf2awN69Z3TQAYxNSo4g5q6BHGEZYKOtClJM2dcn9lgtkngIZ55xhR6cqC3BHdbbY7afq9lOHFGL0qbvDUNT9Ub5X4o72dWV_1M69_OOHyKzt-HN3Z250Tlv98JtlkGyBYzM2_IcIS0PXKFO0-o2Cw5ZIuAecztfVNfA13SpiiUZhX9qIViDgU1X20foYuF_8lBa0J8KrZnA3oJUtyw6pBzdnpL-O21nj47QjLtNT3KCqFA0K8gNQXBYNC794pHdMRjOwY_JWYF-bzBi1FXiOP8FGjFBvsZNW3mH_c27uPkChue0xdzvxlblXjK_CVpZTwhFZYDPDKJvRhoc7qgbnfBrUll0xEhVMWs9EF3dSNIQAGokjUGYYkrOHRJIh5A7DMibG52jz-efGSsc_p97Yh1AfjyR0BrVcig2Uj1nGAK40yrYipO0qTecN7ac15pPU6NrfLteQ0ivooCJ7g5mSlqX4zcADjFJEvX0RmzezbSZBa1qPAkvKt2m80ZqgtJWuUyXgvFfGm0mhswrRzi1Uwd9_U3jGdzS6gpoDLR78uLkSJwXjZDNAqTnJF8ODGLSzDECibzOJCRlQ9RiuWVfvZUq00y1Aplfd2vUelhuY9SSiTJ3j8eXe0zg2XOKs187QjQptSbF_mQzizDdNr7aiS-sHJIWHsxGA9Dh9OLDHd0zQ_H4Cxs-kGpOpa4htQO38cd102cxv-BNctrxPsrumAOvpby2QanFigCvbii6ywOs&uuid=937176ce-abbf-408f-943d-e5742669c423&authuser=0&nonce=6ho4oreev4rv6&user=02343706292923644565&hash=rph8gdbfv45cll7fs6sqka6jitbhlbl1) and put the downloaded `CrossSum_v1.0.tar.bz2` into `data` and unpack it.
2. Generate parallel monolingual summaries by running `bash ./data_preprocess/generate_parallel_xs.sh`.

## Run Experiment

### Preparation

```sh
pip3 install -r requirements.txt
mkdir output_dirs
```

### WikiLingua

```sh
bash ./scripts/wiki/run_korean.sh
bash ./scripts/wiki/run_hindi.sh
bash ./scripts/wiki/run_czech.sh
bash ./scripts/wiki/run_turkish.sh
```

### GlobalVoice

```sh
bash ./scripts/gv/run_french.sh
bash ./scripts/gv/run_arabic.sh
```

### CrossSum

```sh
bash ./scripts/cross/run_french.sh
bash ./scripts/cross/run_arabic.sh
```
