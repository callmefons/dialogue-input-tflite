package jp.sozolab.gtolog.service;


import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import androidx.fragment.app.FragmentActivity;

import com.atilika.kuromoji.unidic.Token;
import com.atilika.kuromoji.unidic.Tokenizer;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jp.sozolab.gtolog.UI.dialog.ActDialogViewPager;
import jp.sozolab.gtolog.activity.MainActivity;
import static jp.sozolab.gtolog.Utility.Utility.log;

public class DialogueClassifier {


    private FragmentActivity fragmentActivity;
    private final Map<String, Integer> dic = new HashMap<>();
    private static final int SENTENCE_LEN = 53; // The maximum length of an input sentence.
    private static final int NUM_CLASSES = 8; // The number of classes.

    private static final String UNKNOWN = "<UNKNOWN>";
    private static final String MODEL_PATH = "text_classification.tflite";
    private Interpreter tflite;
    String[] labels = {"最高血圧(mmHg)"};


    public DialogueClassifier(FragmentActivity fragmentActivity) {
        this.fragmentActivity = fragmentActivity;
    }

    private String loadJSONFromAsset() {
        String json = null;
        try {
            InputStream is = fragmentActivity.getAssets().open("vocab.json");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;
    }

    /** Load the TF Lite model and dictionary so that the client can start classifying text. */
    public void run() {
        loadModel();
    }

    /** Load TF Lite model. */
    private synchronized void loadModel() {
        try {
            // Load the TF Lite model
            ByteBuffer buffer = loadModelFile(fragmentActivity.getAssets(), MODEL_PATH);
            tflite = new Interpreter(buffer);
            log("DialogueClassifier TFLite model loaded.");

            // Use metadata extractor to extract the dictionary and label files.
            MetadataExtractor metadataExtractor = new MetadataExtractor(buffer);

            // Extract and load the dictionary file.
            loadDictionaryFile();
            log("DialogueClassifier Dictionary loaded.");

            // Extract and load the label file.
//            InputStream labelFile = metadataExtractor.getAssociatedFile("labels.txt");
            //loadLabelFile(labelFile);
            //log("Labels loaded.");

        } catch (IOException ex) {
            log("DialogueClassifier Error loading TF Lite model.\n" + ex);
        }
    }

    /** Load TF Lite model from assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
            throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }


    private void loadDictionaryFile(){
        try {
            JSONObject jsonObject = new JSONObject(loadJSONFromAsset());
            JSONArray keys = jsonObject.names ();
            for (int i = 0; i < keys.length(); i++) {
                String key = keys.getString(i);
                Integer value = jsonObject.getInt(key);
                dic.put(key , value);
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    private float[][] tokenizeInputText(String text) {
        List<String> words = new ArrayList<>();
        float[] tmp = new float[SENTENCE_LEN];

        Tokenizer tokenizer = new Tokenizer() ;
        List<Token> tokens = tokenizer.tokenize(text);
        for (Token token : tokens) {
//            log("DialogueClassifier tokenized text: " + token.getSurface());
            words.add(token.getSurface());
        }

        int index = 0;
        for (String word: words) {
            if (word.trim() != ""){
                if (index >= SENTENCE_LEN) {
                    break;
                }
                tmp[index++] = dic.containsKey(word) ? dic.get(word) : (int) dic.get(UNKNOWN);
            }
        }
        Arrays.fill(tmp, index, SENTENCE_LEN - 1, 0);
        float[][] ans = {tmp};

        return ans;
    }

    public synchronized String[] classify(String text){

        float[][] inputs = tokenizeInputText(text);
        float[][] outputs = new float[1][NUM_CLASSES];
        tflite.run(inputs, outputs);

//        log("DialogueClassifier input " + Arrays.deepToString(inputs));
//        log("DialogueClassifier outputs  " + Arrays.deepToString(outputs));

       String[] results = new String[NUM_CLASSES];

        for (int i = 0; i < NUM_CLASSES; i++) {
            String result = "No";
            Integer predicted = Math.round(outputs[0][i]);
            log("DialogueClassifier predicted " + predicted);
            if(predicted == 1) result = "Yes";
            results[i] = result;
        }

        return results;
    }


}
