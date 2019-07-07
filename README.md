# Causeway causal language tagger

Causeway is a system for detecting explicit causal relations in text. It tags text using the BECAUSE 1.0 annotation scheme, described in [Dunietz et al., 2015](http://www.cs.cmu.edu/~jdunietz/publications/causal-language-annotation.pdf). The system itself is described in [Dunietz et al., 2017](http://www.cs.cmu.edu/~jdunietz/publications/causeway-system.pdf).

Note that the repository includes some code for reading in data in an updated version of the annotation scheme ([BECAUSE 2.x](https://www.cs.cmu.edu/~jdunietz/publications/because-v2.pdf)). This newer scheme is backwards-compatible with the original.

The steps to reproduce the results from the 2017 Causeway paper are given below. If you have any difficulty doing so or have additional questions, please [contact Jesse Dunietz](mailto:jdunietz@cs.cmu.edu), who will be happy to assist.

**NOTE:** You may also be interested in [DeepCx](https://github.com/duncanka/lstm-causality-tagger), a neural network tagger that supersedes Causeway. DeepCx achieves substantially better performance on all versions of the BECAUSE dataset.


## Running the tagger
To reproduce the results from the Causeway paper:

1. You'll want to do this in Ubuntu, the only platform Causeway has been tested on. It may work on other *nix platforms, but you'll be on your own for getting it to do so.

   You'll need some standard Ubuntu packages, which you can install using `apt` if you don't have them:
   ```bash
   sudo apt install git python2 python-pip sed task-spooler default-jdk # or any JDK
   ```

2. Install the external Python packages that Causeway depends on:
   ```bash
   sudo pip2 install bidict colorama nltk cython python-gflags numpy scipy scikit-learn python-crfsuite
   ```

3. Clone the Causeway repository, including the [NLPypline](https://github.com/duncanka/NLPypline) framework for NLP pipelines (included as a Git submodule):

   ```bash
   git clone --recursive https://github.com/duncanka/Causeway.git
   ```

4. Set up version 3.5.2 of the Stanford parser.
   1. Download the full Stanford parser [package](https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip). Unzip it somewhere, resulting in a folder called `stanford-parser-full-2015-04-20`.

   2. Within `stanford-parser-full-2015-04-20`, unzip the English PCFG model:
      ```bash
      unzip stanford-parser-3.5.2-models.jar edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz
      ```

   3. Apply the [Causeway-specific patches](../master/stanford-patches) to the Stanford parser. The following hacky script should do the trick (run from the root `stanford-parser-full-2015-04-20` directory; replace `$CAUSEWAY_DIR` with the path of the `Causeway` directory from step 3):
      ```bash
      mkdir /tmp/stanford-sources
      unzip stanford-parser-3.5.2-sources.jar -d /tmp/stanford-sources
      cp $CAUSEWAY_DIR/stanford-patches/*.patch /tmp/stanford-sources
      (cd /tmp/stanford-sources && {
          for PATCH in *.patch; do
              patch -p 2 < $PATCH
          done
      })
      TO_RECOMPILE=$(grep '+++' /tmp/stanford-sources/*.patch | sed -e 's/.*\(edu.*\.java\).*/\1/' | sort | uniq)
      for SRC_FILE in $TO_RECOMPILE; do
          javac -cp /tmp/stanford-sources /tmp/stanford-sources/$SRC_FILE
          jar uf stanford-parser.jar -C /tmp/stanford-sources/ ${SRC_FILE%.java}.class
      done

      rm -R /tmp/stanford-sources

      ```
      You might see a bit of error output from the Java compiler. Don't worry about it.

   4. Download the TRegex/TSurgeon package and extract the relevant files. (The TRegex/TSurgeon binaries are included with the Stanford parser; this is just to get the run scripts.)
      ```bash
      wget https://nlp.stanford.edu/software/stanford-tregex-2018-10-16.zip
      unzip -j stanford-tregex-2018-10-16.zip stanford-tregex-2018-10-16/tregex.sh stanford-tregex-2018-10-16/tsurgeon.sh -d stanford-parser-full-2015-04-20
      rm stanford-tregex-2018-10-16.zip
      ```

5. Reconstitute the [BECAUSE](https://github.com/duncanka/BECauSE) 1.0 corpus.
   1. Clone the repository.
      ```bash
      git clone https://github.com/duncanka/BECAUSE.git
      (cd BECAUSE && git checkout 1.0)
      ```
      We'll refer to the resulting directory named `BECAUSE` as `$BECAUSE_DIR`.

   2. If you haven't previously done so, follow the [NLTK instructions](https://www.nltk.org/data.html#manual-installation) for manually setting up NLTK access to the Penn Treebank. (You'll need to use your institution's LDC license to acquire the data, and combine all the `.mrg` files into one directory.)

   3. Run the WSJ text extraction script to get the raw text for the PTB annotations:
      ```bash
      python $BECAUSE_DIR/scripts/extract_wsj_txt.py $BECAUSE_DIR/PTB
      ```
      You should end up with a bunch of `.txt` files alongside the `.ann` files in the `PTB` subdirectory.

   4. Run the NYT text extraction script on your LDC-licensed copy of the [NYT corpus](https://catalog.ldc.upenn.edu/LDC2008T19), which let's assume is stored in directory `$NYT_DIR`:
      ```bash
      python $BECAUSE_DIR/scripts/extract_nyt_txt.py $BECAUSE_DIR/NYT $(for FNAME in $BECAUSE_DIR/NYT/*.ann; do find $NYT_DIR -name $(basename "${FNAME%.ann}.xml"); done)
      ```
      Again, you should end up with a bunch of `.txt` files alongside the `.ann` files in the `NYT` subdirectory.

6. Run the Stanford parser (located in `$STANFORD_DIR`) on the data:
   ```bash
   for DATA_DIR in $BECAUSE_DIR/PTB $BECAUSE_DIR/NYT $BECAUSE_DIR/CongressionalHearings; do
       $CAUSEWAY_DIR/scripts/preprocess.sh $DATA_DIR $STANFORD_DIR;
   done
   ```

7. While you're waiting, compile the one Cython file in the project:
   ```
   (cd $CAUSEWAY_DIR/NLPypline/src/nlpypline/util && cythonize -i streams.pyx)
   ```

8. Run the system.
   1. Edit the `DATA_DIR`, `PTB_DIR`, and `STANFORD_DIR` variables in [`run_all_pipelines.sh`](scripts/run_all_pipelines.sh) to match your setup.
   2. Run the script from the root Causeway directory.

#### Citations

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. Automatically Tagging Constructions of Causation and Their Slot-Fillers. In press; to be published in 2017. *Transactions of the Association for Computational Linguistics*.</sub>

<sub>Dunietz, Jesse, Lori Levin, and Jaime Carbonell. Annotating Causal Language Using Corpus Lexicography of Constructions. *Proceedings of LAW IX – The 9th Linguistic Annotation Workshop* (2015): 188-196.</sub>
