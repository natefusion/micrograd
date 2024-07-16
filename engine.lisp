(defpackage :engine
  (:use :cl)
  (:shadow
   :* :+ :- :/ :expt :tanh)
  (:export
   :* :+ :- :/ :expt :tanh :letvalue* :backward :draw-tree* :value :relu)
  )

(in-package :engine)

;; (ql:quickload "gtfl")
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
    :initarg :name
    :initform nil)
   (local-grads
    :initarg :local-grads
    :initform nil)))

(defmethod initialize-instance :after ((obj value) &key))

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

(defgeneric + (a b)
  (:method ((a value) (b value))
    (make-instance
     'value
     :data (cl:+ (data a) (data b))
     :children (list a b)
     :local-grads (list 1 1)
     :op '+))
  (:method ((a value) (b number)) (+ a (make-instance 'value :data b)))
  (:method ((a number) (b value)) (+ (make-instance 'value :data a) b)))

(defgeneric * (a b)
  (:method ((a value) (b value))
    (make-instance
     'value
     :data (cl:* (data a) (data b))
     :children (list a b)
     :local-grads (list (data b) (data a))
     :op '*))
  (:method ((a number) (b value)) (* (make-instance 'value :data a) b))
  (:method ((a value) (b number)) (* a (make-instance 'value :data b))))

(defun - (a b) (+ a (* b -1)))
(defun / (a b) (* a (expt b -1)))

(defgeneric tanh (a)
  (:method ((a value))
    (let ((tanh (cl:tanh (data a))))
      (make-instance
       'value
       :data tanh
       :children (list a)
       :local-grads (list (cl:- 1 (cl:expt tanh 2)))
       :op 'tanh))))

(defgeneric expt (base power)
  (:method ((base value) (power number))
    (make-instance
     'value
     :data (cl:expt (data base) power)
     :children (list base)
     :local-grads (list (cl:* power (cl:expt (data base) (1- power))))
     :op (read-from-string (format nil "^~a" power)))))

(defgeneric relu (a)
  (:method ((a value))
    (make-instance
     'value
     :data (if (> (data a) 0) (data a) 0)
     :children (list a)
     :local-grads (list (if (> (data a) 0) 1 0))
     :op 'relu)))

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
      (setf (grad a) 1)
      (dolist (v topo)
        (loop for child in (slot-value v 'children)
              for local-grad in (slot-value v 'local-grads)
              do (incf (grad child) (cl:* local-grad (grad v))))))))

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


