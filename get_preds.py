def get_predictions(X, saved_model_loc, batch_size = 32):
    graph = tf.get_default_graph()
    with tf.Session(graph = graph) as sess:
        # Restore saved values
        
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], saved_model_loc)
        
        # Get restored placeholders
        inputs = graph.get_tensor_by_name('inputs_ph:0')

        # Get restored model output
        predictions = graph.get_tensor_by_name('predictions:0')

        m = X.shape[0]
        num_batches = m//batch_size + 1

        Y_preds = np.empty(shape = m)
        
        for batch in range(num_batches):
                if batch != num_batches - 1:
                    batch_index = range(batch*batch_size, (batch+1)*batch_size)
                else:
                    batch_index = range(batch*batch_size,m)
                X_batch = X[batch_index]
                Y_batch_preds = sess.run(predictions, feed_dict ={inputs: X_batch})
                Y_preds[batch_index] = Y_batch_preds
    return Y_preds



    saved_model_loc = model+'-'+ts