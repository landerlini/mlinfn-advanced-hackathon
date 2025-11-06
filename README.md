# Material for the AI-INFN Advanced Hackathon (Pavia version)

AI-INFN, as evolution of the ML-INFN initiative, collects and coordinate the efforts 
on the development and deployment of Artificial Intelligence technologies relevant 
to INFN research. As part of our programme, we organize training events to discuss 
base and advanced Machine Learning topics with time to go through the code.
We call them _hackathons_.

The [first](https://agenda.infn.it/event/25855), 
[second](https://agenda.infn.it/event/28565), and 
[fourth](https://agenda.infn.it/event/35607) ML-INFN hackathons were targeting
Machine Learning beginners. The material used for those events is available in 
[another GitHub repository](https://github.com/tommasoboccali/ml_infn_hackBase).

The [third](https://agenda.infn.it/event/32568) and 
[fifth](https://agenda.infn.it/event/37650) ML-INFN hackathons were targeting 
advanced users, as well as this [first](https://agenda.infn.it/event/43129) 

AI-INFN hackathon and the related materials are collected in this repository.
The [first](https://agenda.infn.it/event/43129/) hackathon was held in Padua
in 2024, while the [second](https://agenda.infn.it/event/47736/), is hosted 
in Pavia in 2025.

## Structure of the repository
Contents is organized per topic in different folders. 
When documentation beyond the Jupyter notebook is needed, a README.md file is 
included in the sub-directory.

### Contents
* `ex`: material for the hackathon exercises
* `test`: automated testing infrastructure

## Automated testing
Tests on the notebooks are run frequently on the different setups being prepared
for the hackathon event.

Run all tests with:
```bash
python3 -m pytest tests/test_notebooks.py -v --durations=0
```

## License
Code is released under OSI-approved [MIT license](./LICENSE).

The documentation provided in the form of Jupyter notebooks is 
released under [CC-BY-NC-SA](./CC-BY-NC-SA-4.0) license.
