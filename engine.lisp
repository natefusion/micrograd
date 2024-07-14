(defpackage :engine
  (:use :cl)
  (:shadow
   :* :+ :- :/ :expt :tanh
   ))
(in-package :engine)

(ql:quickload "gtfl")
(gtfl:start-gtfl)

(defclass value ()
  ((data
    :accessor data
    :initarg :data)
   (grad
    :accessor grad
    :initform 0)
   (children
    :initarg :children
    :initform nil)
   (op
    :initarg :op
    :initform nil)
   (name
    :initarg :name)
   (local-grads
    :initarg :local-grads
    :initform nil)))

(defmethod initialize-instance :after ((obj value) &key children)
  (setf (slot-value obj 'children) (remove-duplicates children)))

(defmethod print-object ((obj value) stream) 
  (print-unreadable-object (obj stream :type 't)
    (format stream "data = ~a" (slot-value obj 'data))))

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


(defgeneric + (a b))
(defgeneric * (a b))
(defgeneric - (a &optional b))
(defgeneric / (a b))
(defgeneric tanh (a))
(defgeneric expt (base power))
(defgeneric backward (a))

(defmethod backward ((a value))
  (let ((topo nil)
        (visited nil))
    (labels ((build-topo (v)
               (unless (find v visited)
                 (push v visited)
                 (dolist (child (slot-value v 'children))
                   (build-topo child))
                 (push v topo))))
      (build-topo a)
      (setf (grad a) 1)
      (dolist (v topo)
        (with-slots (children local-grads) v
          (loop for (child . local-grad) in (pairlis children local-grads)
                do (incf (grad child) (cl:* local-grad (grad v)))))))))

(defmethod + ((a value) (b value))
  (make-instance
   'value
   :data (cl:+ (data a) (data b))
   :children (list a b)
   :local-grads (list 1 1)
   :op '+))

(defmethod - ((a value) &optional b)
  (etypecase b
    (null (* a -1))
    (value (+ a (* b -1)))))

(defmethod * ((a value) (b value))
  (let ((result
          (make-instance
           'value
           :data (cl:* (data a) (data b))
           :children (list a b)
           :local-grads (list (data b) (data a))
           :op '*)))
    result))

(defmethod * ((a number) (b value))
  (* (make-instance 'value :data a) b))

(defmethod * ((a value) (b number))
  (* a (make-instance 'value :data b)))

(defmethod / ((a value) (b value))
  (* a (expt b -1)))

(defmethod / ((a number) (b value))
  (* (make-instance 'value :data a) (expt b -1)))

(defmethod / ((a value) (b number))
  (* a (expt (make-instance 'value :data b) -1)))

(defmethod expt ((base value) (power number))
  (make-instance
   'value
   :data (cl:expt (data base) power)
   :children (list base)
   :local-grads (list (cl:* power (cl:expt (data base) (1- power))))
   :op (read-from-string (format nil "^~a" power))))

(defmethod tanh ((a value))
  (let ((tanh (cl:tanh (data a))))
    (make-instance
     'value
     :data tanh
     :children (list a)
     :local-grads (list (cl:- 1 (cl:expt tanh 2)))
     :op 'tanh)))

(defmacro letvalue* ((&rest bindings) &body body)
  (loop for x in bindings
        collect (car x) into vars
        append `((slot-value ,(car x) 'name) ',(car x)) into setfs
        finally (return `(let* ,bindings
                           (declare (value ,@vars))
                           (setf ,@setfs)
                           ,@body))))

(letvalue* ((x1 (make-instance 'value :data 2))
            (x2 (make-instance 'value :data 0))
            (w1 (make-instance 'value :data -3))
            (w2 (make-instance 'value :data 1))
            (b  (make-instance 'value :data 6.8813735870195432))
            (x1*w1 (* x1 w1))
            (x2*w2 (* x2 w2))
            (x1*w1+x2*w2 (+ x1*w1 x2*w2))
            (n (+ x1*w1+x2*w2 b))
            (o (tanh n)))
  (backward o)
  (gtfl:reset-gtfl)
  (gtfl:gtfl-out (draw-tree* o)))

(letvalue* ((a (make-instance 'value :data 3))
            (b (+ a a)))
  (backward b)
  (gtfl:reset-gtfl)
  (gtfl:gtfl-out (draw-tree* b)))


