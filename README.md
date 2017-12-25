If you have chance look into this [Kaggle Competition][1], or if you have deeped dive into applying deep learning on sound signal processing,
you may have patience to read this post. Otherwise, you may feel the post is resident, because I'm going to record my experiments of this 
Competition in detail.

<h2 id = 'agenda'>Agenda</h2>


<h2 id = 'overview'>Overview</h2>
<h2 id = 'Data'>Look into dataset</h2>
<h3 id = 'sound_basic'>Sound basic theory</h3>
<h3 id ='feature'>Feature map</h3>

<h2 id ='Strategy'>Strategy</h2>
<h3 id = 'strategy1'>Tuning the parameters of the tutorial given by Google</h3>
Train the model as the tutorial, the accurancy rate will be around 90%.
{% highlight shell linenos %}
nohup python tensorflow/examples/speech_commands/train.py \
--how_many_training_steps=25000,25000 \
--learning_rate=0.001,0.0001 \
--train_dir=/home/maikfangogoair/tmp/save/tutorial \
--model_architecture=conv > /home/maikfangogoair/tmp/save/tutorial/conv.log &
{% endhighlight %}
Freeze the model:
{% highlight shell linenos %}
python tensorflow/examples/speech_commands/freeze.py \
--start_checkpoint=/home/maikfangogoair/tmp/save/tutorial/conv.ckpt-50000 \
--output_file=/home/maikfangogoair/tmp/save/tutorial/conv_graph.pb
{% endhighlight %}
Test the model, as the tutorial shows, this [label_wav.py][6] can evaluate one wave file one time. 
{% highlight shell linenos %}
python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav
{% endhighlight %}
I'm going to slightly change the python file and make [label_wav_multiple.py] to evaluate all the wild test data
and generate submission file for the competition directly. It will take around 30 mins to evaluate.
{% highlight shell linenos %}
nohup python label_wav_multiple.py \
--graph=/home/maikfangogoair/tmp/save/tutorial/conv_graph.pb \
--labels=/home/maikfangogoair/tmp/save/tutorial/conv_labels.txt \
--wave_dir=/home/maikfangogoair/test/audio/ \
--submission_file=/home/maikfangogoair/tmp/save/tutorial/submission.csv > nohup.out &
{% endhighlight %}
There is a trap in the competition. If you look carefully into the example submission file,
you will find the "silence" and "unknown" label are different in the program. So replace them and submit.
{% highlight shell linenos %}
cd /home/maikfangogoair/tmp/save/tutorial/
sed -e 's/_silence_/silence/g' -e 's/_unknown_/unknown/g' submission.csv > submission_final.csv
{% endhighlight %}
To start the competition, it's a good entry to try on the [tutorial][2] given by Google.
Actually the tutorial is not only an entry, but also give a lot background about 
the state-of-art technique on sound signal processing and tricky method to everlage 
Tensorflow framework and uniform performance measurement stragety on embedded devices.
If you check the tutorial, you will surely look into the [source code][3] and naturally 
find it's a robust model and highly coupling internally. So it's unfriendly to beginner
like me to split some useful code snippet from it, because I want to buidl an end-to-end
model, so that I can benchmark from features to models by myself, and I believe it's necessary
to do so to understand the competition/dataset entirely and get the good scores in the competition.
If you still want to integrate your ideas into the tutorial codes, 
maybe the [implementation][7] of [this paper][8] will be helpful, which benchmark performances of various
models on Raspberry Pi. So it's also a good entry for the [special prize competition](#raspberry).

<h3 id = 'strategy2'>Start from some existing model or baseline</h3>
As the Google's tutorial is too tricky for me to de-couple, I try to seek for some other resource implementation.
Luckily, I found a Pytorch version implementation by [VGGnet]() for the Google's tutorial , which is concise and friendly
to beginners. The most attractive thing is it achieve 94% testing accurancy rate on the labelled dataset. 
So I plan to take it as my baseline to start my research. This implementation is a classic supervised deep learning model
to train the model learning 30 classes of sound. It use the same training dataset, but without the "unknown" and "silence"
segmentation.   
Therefore, there are two main gaps between the 99.4% accurancy implementation and the competition requirements :   
1. The implementation targets 30 classes and doesn't train for "silence" and "unknown". 
    The competition class is ```['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']```  
2. The implementation test on the testing data from the labeled dataset. But the competition will compete on the wild testing sounds.
    There must be some difference between these two datasets, and this is the main difficulty of the competition.
    I suppose the two datasets differ in: a) the labeled data is higher data quality than the wild testing data.
    b) the wild testing data may have data out of the classification of the labeled data.  
Suppose I can successfully make the same model on Tensorflow and gain more than ~99% testing accurancy on labeled data. 
And for the wild testing, I can take all classes not in the competition required as "unknown". And suppose that if the model 
is already robust, then I will only miss the "silence" class for the competition, which maybe at most account for 10 percentage 
of the wild test data. Then the ideal ~90% wild testing accurancy will let me win the competition easily. 
(Now the highest score is 89% accurancy rate) But there must be some gap between the datasets and tricks left by Google for the competitors.  
```Adding implementation here```  
The wild testing accurancy is 71%, which is much lower than my ideal suppose before. But it's a good baseline for me to improve the model
based on several points:  
1. Train the model to figure out silence as much as possible. If you can't wait to know this solution, you can [skip to here](#silence).  
2. Train the model to understand what is realy "unknown" instead of pretend to be "unknown" on the known classes. Because there must 
some wild data out of the scope of the training data, which should be identify as "unknown" for the model. You can also [skip to here](#unknown).   
3. Intergrate the model with state-of-art "data augmentation" theory to rebust the model and avoid overfitting, because as my suppose before,
the wild test dataset must be in lower quality than the labeled data, and besides, the wild test dataset is much more larger than the labeled dataset.
Please skip to [here](#data_augmentation) for detail.   

<h3 id = 'silence'>Compete with the "silence" class</h3>
[As the tutorial][9], silence training data is created from the sound snippets of the ```_background_noise_``` folder.  
Like the unkown data, raising the percentage of silence data in the training dataset will lead to higher possibility of predicting 
silence. My understanding is "silence" is not absolutely silent, and there must be some background noise in nature. So the "silence" 
can be identify as no human being voices in the sound snippets, the background noise is acceptable and maybe very loud. The intuition 
is the sound wav for "silence" is continuous.
<h3 id = 'unknown'>Compete with the "unknown" class</h3>
<h3 id = 'data_augmentation'>Data Augmentation</h3>

<h2 id ='benchmark'>Benchmark</h2>
<h3 id =''>Features</h3>
<h3 id =''>Models</h3>

<h2 id ='raspberry'>Test on Raspberry Pi 3 for the special prize</h2>
<h3 id = 'rule'>Rule for the special prize<h3>
<h3 id = 'benchmark_rpi'>Benchmark<h3>

<h2 id ='conclusion'>Conclusion</h2>

<h2 id ='reference'>Reference</h2>

[1]: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/
[2]: https://www.tensorflow.org/tutorials/audio_recognition
[3]: 
[6]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/label_wav.py
[7]: https://github.com/ARM-software/ML-KWS-for-MCU
[8]: https://arxiv.org/pdf/1711.07128.pdf
[9]: https://www.tensorflow.org/tutorials/audio_recognition#silence


