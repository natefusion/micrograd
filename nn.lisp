(defclass neuron ()
  (w
   (b :initform (make-instance 'engine:value :data 0))
   (nonlin :initarg :nonlin :initform nil)))

(defclass layer () ((neurons :initform nil)))
(defclass mlp () ((size :initarg :dimensions) layers))

(defmethod initialize-instance :after ((obj neuron) &key nin layer neuron)
  (setf (slot-value obj 'w)
        (loop for i below nin
              collect (make-instance 'engine:value
                                     :name (read-from-string (format nil "w~a~a~a" i neuron layer))
                                     :data (1- (random 2.0))))
        (slot-value (slot-value obj 'b) 'engine::name) (read-from-string (format nil "b~a~a" neuron layer))))

(defmethod initialize-instance :after ((obj layer) &key nin nout nonlin layer)
   (setf (slot-value obj 'neurons)
        (loop for i below nout
              collect (make-instance 'neuron :neuron i :layer layer :nin nin :nonlin nonlin))))

(defmethod initialize-instance :after ((obj mlp) &key dimensions)
  (setf (slot-value obj 'layers)
        (loop for i below (1- (length dimensions))
              collect (make-instance
                       'layer
                       :layer i
                       :nin (nth i dimensions)
                       :nout (nth (1+ i) dimensions)
                       :nonlin (/= i (- (length dimensions) 2))))))

(defmethod print-object ((obj neuron) stream)
  (print-unreadable-object (obj stream :type 't)
    (format stream "~a ~a"
            (if (slot-value obj 'nonlin) "ReLU" "Linear")
            (length (slot-value obj 'w)))))

(defmethod print-object ((obj layer) stream)
  (print-unreadable-object (obj stream :type 't)
    (format stream "of ~{~a~^,~}"
            (slot-value obj 'neurons))))

(defmethod print-object ((obj mlp) stream)
  (print-unreadable-object (obj stream :type 't)
    (format stream "of ~{~a~^,~}"
            (slot-value obj 'layers))))

(defgeneric forward (a x)
  (:method ((a neuron) (x list))
    (loop for wi in (slot-value a 'w)
          for xi in x
          for i from 0
          with act = (slot-value a 'b)
          do (setf act (engine:+ act (engine:* wi xi)))
          finally (return (if (slot-value a 'nonlin) (engine:relu act) act))))
  (:method ((a layer) (x list))
    (loop for n in (slot-value a 'neurons)
          collect (forward n x) into out
          finally (return (if (cdr out) out (car out)))))
  (:method ((a mlp) (x list))
    (dolist (layer (slot-value a 'layers) x)
      (setf x (forward layer x)))))

(defgeneric parameters (a)
  (:method ((a mlp))
    (loop for layer in (slot-value a 'layers) append (parameters layer)))
  (:method ((a layer))
    (loop for n in (slot-value a 'neurons) append (parameters n)))
  (:method ((a neuron))
    (append (slot-value a 'w) (list (slot-value a 'b)))))

(defun zero-grad (a)
  (dolist (p (parameters a))
    (setf (slot-value p 'engine::grad) 0)))

(let ((mlp (make-instance 'mlp :dimensions (list 3 4 4 1)))
      (xs (list (list 2 3 -1) (list 3 -1 .5) (list .5 1 1) (list 1 1 -1)))
      (ys (list 1 -1 -1 1))
      loss)
  (loop repeat 10 do
    (let ((ypred (loop for x in xs collect (forward mlp x))))
      (engine:letvalue* ((l (loop for ygt in ys
                                  for yout in ypred
                                  with sum = 0
                                  do (setf sum (engine:+ sum (engine:expt (engine:- yout ygt) 2)))
                                  finally (return sum))))
        (setf loss l)
        (engine:backward l)
        (dolist (p (parameters mlp))
          (incf (engine::data p) (* -0.01 (engine::grad p)))))))
  (gtfl:reset-gtfl)
  (gtfl:gtfl-out (engine:draw-tree* loss)))
