# Contributing Guidelines
This contrib is seeking examples that will accelerate the use of Matlantis. Examples include:
- Visualizing the target atomic system in an easy-to-understand manner
- Aggregating and analyzing the execution results of features implemented in Matlantis
- Creating input examples for features implemented in Matlantis

These are just a few examples, and we welcome examples on a wide range of topics. Also, as long as the content is useful for performing atomic simulations, it is not necessary to import Matlantis into the notebook.

## Before Contributing
- The programs in this repository are licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). To participate in this project, you must review and agree to the terms of the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- The copyright of the examples belongs to their authors, but the copyright of the entire project is held by `Copyright 2023 Matlantis Contributors`. We will write it as follows.
- A signature is required for the [DCO](https://github.com/probot/dco#how-it-works). You can sign the DCO by adding the `--signoff` or `-s` option when executing `git commit`.
- Particularly versatile examples added to contrib may be added to the examples and tutorials of [Matlantis](https://matlantis.com/en/).
- If you notice a bug in an example in contrib, please file an issue in [Issues](https://github.pfidev.jp/Matlantis/matlantis-contrib/issues) in this repository rather than submitting a pull request directly.

## Example Format
When submitting a pull request to add an example, please make sure that the following conditions are met. Reviewers will primarily review the project to ensure that these conditions are met.
- The DCO must be signed.
- No errors occur when running. Run the notebook's cells in the Matlantis environment from top to bottom until no errors occur.
- Include a copyright notice. To clearly indicate that the program creator owns the copyright, add the following statement to the top of the notebook in Markdown format: ``Copyright <YOUR NAME> as contributors to Matlantis contrib project```. Replace `<YOUR NAME>` with the name of the program creator. Refer to the contents of [hello_world](matlantis_contrib_examples/hello_world/hello_world.ipynb).
- Do not include confidential information or authentication information (API keys or passwords).

## Guidelines
### Forking the repository
First, fork matlantis-contrib on GitHub. For instructions on how to fork, refer to the official documentation. After forking the repository, clone it to your computer as follows:
```
git clone git@github.com:YOUR_NAME/matlantis-contrib.git
cd matlantis-contrib
```
### Branch Naming
The branch name should be the same as the name of the example you are adding. For example, to add an example called `a_great_example`, create the branch as follows:
```
git checkout -b a_great_example
```
### Directory Structure
Each example corresponds to a directory in `matlantis_contrib_examples`. To add an example called `a_great_example`, create the following structure:
```
matlantis_contrib_examples
└── a_great_example(directory)
├── a_great_example.ipynb
├── input(directory)
| ├── hoge.xyz
| └── fuga.xyz
└── output(directory)
└── piyo.xyz
```
If `a_great_example.ipynb` has input/output files, place them in the input/output directory. If there are no input/output files, omit creating the input/output directories. Adding output files is optional. It is recommended not to add output files, especially if the output file is large (over 1MB), such as an MD trajectory file. Use relative paths within the notebook so that files are correctly input/output to the input/output directory when `a_great_example.ipynb` is executed. For details, refer to [hello_world](matlantis_contrib_examples/hello_world/hello_world.ipynb).
### DCO Signature
DCO signature is required when executing `git commit`. DCO signature can be performed as follows:
```
git commit --signoff -m "add a_great_example"
```
or
```
git commit -s -m "add a_great_example"
```
Don't forget to add the `--signoff` or `-s` option when committing.
### Before submitting a pull request,
Please make sure it follows the [example format](#example format).