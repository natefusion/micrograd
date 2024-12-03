(defpackage :engine
  (:use :cl)
  (:shadow
   :* :+ :- :/ :expt :tanh)
  (:export
   ;; Don't specify symbols to export using keywords. This puts copies of
   ;; those symbols into the keyword package as well, wasting space.
   :* :+ :- :/ :expt :tanh :letvalue* :backward :draw-tree* :value :relu))

(in-package :engine)

;; (ql:quickload "gtfl")
(gtfl:start-gtfl)

(defclass tensor ()
  ((data
    :initarg :data
    :type (simple-array))
   (local-grads
    :initarg :local-grads
    :type (vector (simple-array)))
   (grad
    :type (simple-array))
   (requires-grad
    :type boolean)
   (children
    :initarg :children
    :type (vector 'tensor))
   (op
    :initarg :op
    :type symbol
    :initform nil)
   (name
    :initarg :name
    :type symbol
    :initform nil)))

(defmethod print-object ((obj tensor) stream) 
  (print-unreadable-object (obj stream :type 't)
    (format stream "~a has data ~a" (slot-value obj 'name) (slot-value obj 'data))))

(defun draw-node (string) 
  (gtfl:who 
   (:div :style "font-family:monospace;padding:4px;border:1px solid #888;margin-top:4px;margin-bottom:4px;background-color:#eee;"
         (princ string))))

(defun draw-tree* (value &optional (parent value)) 
  (with-slots (children op data grad name) value
    (with-slots ((p-name name)) parent
        (gtfl:draw-node-with-children
         ;;                                                              wft??
         (gtfl:who-lambda (draw-node (format nil "[~a=~a, ∂~a/∂~a=~a]~:[~; &lt;- ~a~]" name data p-name name grad op op)))
         (mapcar (lambda (x) (gtfl:who-lambda (draw-tree* x parent))) children)))))

(declaim (ftype (function (tensor tensor) (values tensor &optional)) + * - /)
         (ftype (function (tensor) (values tensor &optional)) tanh relu)
         (ftype (function (tensor) null) backward)
         (ftype (function (tensor number)) expt))

(defmacro array-map ((var dimensions) &body body)
  `(loop for ,var below (apply #'cl:* ,dimensions)
         with result = (make-array ,dimensions)
         do (setf (row-major-aref result ,var) ,@body)
         finally (return result)))

(defun tensor-op (op t1 t2)
  (let ((dim-t1 (array-dimensions t1))
        (len-t1 (array-total-size t1))
        (dim-t2 (array-dimensions t2))
        (len-t2 (array-total-size t2)))
    (cond ((= 1 len-t1)      (array-map (i dim-t2) (funcall op (row-major-aref t2 i) (row-major-aref t1 0))))
          ((= 1 len-t2)      (array-map (i dim-t1) (funcall op (row-major-aref t1 i) (row-major-aref t2 0))))
          ((= len-t1 len-t2) (array-map (i dim-t1) (funcall op (row-major-aref t1 i) (row-major-aref t2 i))))
          (t (error "Cannot add TENSOR A that has shape ~a with TENSOR B that has shape ~a" dim-t1 dim-t2)))))

(defun + (a b)
  (let ((new (tensor-op #'cl:+ (slot-value a 'data) (slot-value b 'data))))
    (make-instance
     'tensor
     :data new 
     :children (vector a b)
     :local-grads
     (vector (make-array (array-dimensions new) :initial-element 1)
             (make-array (array-dimensions new) :initial-element 1))
     :op '+)))

(defun * (a b)
  (let ((new (tensor-op #'cl:* (slot-value a 'data) (slot-value b 'data))))
    (make-instance
     'tensor
     :data new
     :children (vector a b)
     :local-grads (vector (slot-value b 'data) (slot-value a 'data))
     :op '*)))

(defun - (a b) (+ a (* (make-instance 'tensor :data #(-1)) b)))
(defun / (a b) (* a (expt b (make-instance 'tensor :data #(-1)))))

(defun tanh (a)
  (with-slots (data) a
    (let ((tanh (array-map (i (array-dimensions data)) (cl:tanh (row-major-aref data i)))))
      (make-instance
       'tensor
       :data tanh
       :children #(a)
       :local-grads
       (vector (array-map (i (array-dimensions data))
                 (cl:- 1 (cl:expt (row-major-aref tanh i) 2))))
       :op 'tanh))))

(defun relu (a)
  (with-slots (data) a
    (make-instance
     'tensor
     :data
     (array-map (i (array-dimensions data))
       (if (> (row-major-aref data i) 0) (row-major-aref data i) 0)) 
     :children #(a)
     :local-grads
     (vector (array-map (i (array-dimensions data))
               (if (> (row-major-aref data i) 0) 1 0)))
     :op 'relu)))

(defun expt (base power)
  (with-slots (data) base
    (make-instance
     'tensor
     :data (array-map (i (array-dimensions data)) (cl:expt (row-major-aref data i) power)) 
     :children #(base)
     :local-grads
     (vector (array-map (i (array-dimensions data))
               (cl:* power (cl:expt (row-major-aref data i) (1- power)))))
     :op (read-from-string (format nil "^~a" power)))))

(defun backward (a)
  (let ((topo nil)
        (visited nil))
    (labels ((build-topo (v)
               (unless (find v visited)
                 (push v visited)
                 (dolist (child (slot-value v 'children))
                   (build-topo child))
                 (push v topo))))
      (build-topo a)
      (dotimes (i (array-total-size (slot-value a 'data)))
        (setf (row-major-aref (slot-value a 'data) i) 1))
      (dolist (v topo)
        (with-slots (children local-grads (v-grad grad)) v
          (loop for child across children
                for local-grad across local-grads
                do (with-slots ((child-grad grad)) child
                     (dotimes (i (array-total-size child-grad))
                       (incf (row-major-aref child-grad i)
                             (cl:* (row-major-aref local-grad i)
                                   (row-major-aref v-grad i)))))))))))


(defmacro letvalue* ((&rest bindings) &body body)
  (loop for x in bindings
        collect (car x) into vars
        append `((slot-value ,(car x) 'name) ',(car x)) into setfs
        finally (return `(let* ,bindings
                           (declare (value ,@vars))
                           (setf ,@setfs)
                           ,@body))))
