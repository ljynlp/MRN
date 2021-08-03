# MRN: A Locally and Globally Mention-Based Reasoning Network for Document-Level Relation Extraction

Source code for ACL-IJCNLP 2021 findings paper: 
[MRN: A Locally and Globally Mention-Based Reasoning Network for Document-Level Relation Extraction](https://aclanthology.org/2021.findings-acl.117/).

## 1. Environments

- python (3.6.8)
- cuda (10.1)

## 2. Dependencies

- numpy (1.17.3)
- torch (1.6.0)
- transformers (3.5.1)
- pandas (1.1.5)
- scikit-learn (0.23.2)

## 3. Preparation

- Download [DocRED](https://github.com/thunlp/DocRED) dataset
- Put all the `train_annotated.json`, `dev.json`, `test.json`,`word2id.json`,`vec.npy`,`rel2id.json`,`ner2id` into the directory `data/`

```bash
>> python preprocess.py
```

## 4. Training

```bash
>> python main.py
```

## 5. Submission to LeadBoard (CodaLab)

You will get json file named `result.json` for test set at step 5. Then you can submit it to [CodaLab](https://competitions.codalab.org/competitions/20717#learn_the_details).

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 7. Citation

If you use this work or code, please kindly cite the following paper:

```bib
@inproceedings{li-etal-2021-mrn,
    title = "{MRN}: A Locally and Globally Mention-Based Reasoning Network for Document-Level Relation Extraction",
    author = "Li, Jingye  and
      Xu, Kang  and
      Li, Fei  and
      Fei, Hao  and
      Ren, Yafeng  and
      Ji, Donghong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.117",
    doi = "10.18653/v1/2021.findings-acl.117",
    pages = "1359--1370",
}
```