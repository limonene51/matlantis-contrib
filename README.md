[日本語版はこちら (Japanese Version)](README.jp.md)

# Matlantis Contrib
This repository is a contrib for useful examples that are helpful when using [Matlantis](https://matlantis.com/en/). By bringing together examples created by Matlantis users, we aim to make Matlantis easier to use.

## Contributing
If you would like to add an example to this contrib, please create a pull request following the [contributing guidelines](CONTRIBUTING.md).

## What kinds of examples are we looking for?
Examples that will accelerate the use of Matlantis include:
- Visualizing the target atomic system in an easy-to-understand manner
- Aggregating and analyzing the execution results of features implemented in Matlantis
- Creating input examples for features implemented in Matlantis

We welcome examples of the above content. Please note that these are just a few examples. If you create useful examples for other content, please add them to this contrib.

## How to use examples
The submitted examples are added to [matlantis_contrib_examples] (matlantis_contrib_examples), and each directory in matlantis_contrib_examples corresponds to one example. The directory structure for each example is as follows:

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
- To run the example `a_great_example`, compress the directory `a_great_example` into a zip file and upload it to Matlantis. Right-click on the file tree pane and select `Extract Archive` to unzip it, then run `a_great_example.ipynb`.
- If an example requires an input file, place the input file in `a_great_example/input`.
- If the example outputs the execution results to a file, the results will be output to `a_great_example/output`.

## Notes

* We do not guarantee the operation of the examples provided in this contrib. They may stop working with updates to Matlantis.
* The contents of this contrib may be modified or deleted in the future without notice.