export ANALYTICS_ZOO_HOME=/home/yang/sources/zoo/dist

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh \
   --master local[4] \
   --driver-memory 10g \
   main.py \
   --dictionaries data/vocab.en.json data/vocab.de.json \
   --datasets data/corpus.en data/corpus.de \
   --valid_source_dataset data/corpus.en
