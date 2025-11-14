import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import VGG16  # as example if needed
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import TFBertModel, BertTokenizer
import lightgbm as lgb

# Import Vision Transformer module (copy vit.py as vit_module.py or similar)
from vit_module import VisionTransformer


def weighted_cross_entropy(y_true, y_pred):
    # Define your weighted cross entropy here or use tf built-in if suitable
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    return loss


def gradient_harmonic_loss(y_true, y_pred):
    # Placeholder for your Gradient Harmonic Loss implementation
    # This requires custom gradients calculations, typically
    loss = tf.reduce_mean(tf.abs(y_true - y_pred))  # example dummy
    return loss


def conv_encoder(input_img):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded


def conv_decoder(encoded):
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(13, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


def proposed_model(X_img_train, X_img_test, X_txt_train, X_txt_test, y_train, y_test, label_encoder, vocab_size,
                   epochs):
    # Image Input
    img_input = Input(shape=(28, 28, 13), name='img_input')
    # Vision Transformer
    vit_model = VisionTransformer(image_size=28, patch_size=7, num_classes=None, dim=64, depth=6, heads=8, mlp_dim=128)
    vit_features = vit_model(img_input)

    # Conv Encoder Decoder for image reconstruction (optional loss)
    encoded_img = conv_encoder(img_input)
    decoded_img = conv_decoder(encoded_img)

    # Text Input
    txt_input = Input(shape=(100,), dtype=tf.int32, name='txt_input')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_outputs = bert_model(txt_input)
    txt_features = bert_outputs.pooler_output  # [batch, hidden_size]

    # Concatenate text and image features
    combined_features = Concatenate()([vit_features, txt_features])

    # Flatten or dense layer for LightGBM input (prepare outside model)
    combined_dense = Dense(128, activation='relu')(combined_features)

    # Model for feature extraction only
    feature_extractor = Model(inputs=[img_input, txt_input], outputs=[combined_dense, decoded_img])

    # Prepare the LightGBM data
    train_features, _ = feature_extractor.predict([X_img_train, X_txt_train], batch_size=32)
    test_features, _ = feature_extractor.predict([X_img_test, X_txt_test], batch_size=32)

    # LightGBM classifier training
    lgb_train = lgb.Dataset(train_features, label=y_train)
    lgb_eval = lgb.Dataset(test_features, label=y_test, reference=lgb_train)

    params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'is_unbalance': True,
        'verbose': -1
    }

    lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=100,
                           early_stopping_rounds=10)

    # Predict and inverse transform labels
    preds_proba = lgbm_model.predict(test_features)
    preds = np.argmax(preds_proba, axis=1)
    preds_labels = label_encoder.inverse_transform(preds)

    # Compute your metrics with preds_labels and y_test decoded similarly
    y_test_labels = label_encoder.inverse_transform(y_test)

    def compute_metrics(y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    metrics = compute_metrics(y_test_labels, preds_labels)

    return feature_extractor, lgbm_model, metrics
