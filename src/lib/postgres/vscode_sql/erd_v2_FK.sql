--Foreign Keys Constraint
-- USERS
ALTER TABLE IF EXISTS public.users
    ADD CONSTRAINT fk_roles_id FOREIGN KEY (roles_id) REFERENCES public.roles (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.users VALIDATE CONSTRAINT fk_roles_id;

ALTER TABLE IF EXISTS public.users
    ADD CONSTRAINT fk_status_id FOREIGN KEY (status_id) REFERENCES public.account_status (id) ON DELETE SET NULL ON UPDATE CASCADE NOT VALID;

ALTER TABLE public.users VALIDATE CONSTRAINT fk_status_id;

-- SESSION_LOG
ALTER TABLE IF EXISTS public.session_log
    ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION NOT VALID;

ALTER TABLE public.session_log VALIDATE CONSTRAINT;

-- PROJECT
ALTER TABLE IF EXISTS public.project
    ADD CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.project VALIDATE CONSTRAINT fk_deployment_id;

-- TRAINING
-- training model
ALTER TABLE IF EXISTS public.training
    ADD CONSTRAINT fk_training_model_id FOREIGN KEY (training_model_id) REFERENCES public.models (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.training VALIDATE CONSTRAINT fk_training_model_id;

-- attached model
ALTER TABLE IF EXISTS public.training
    ADD CONSTRAINT fk_attached_model_id FOREIGN KEY (attached_model_id) REFERENCES public.models (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.training VALIDATE CONSTRAINT fk_attached_model_id;

-- project id
ALTER TABLE IF EXISTS public.training
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.training VALIDATE CONSTRAINT fk_project_id;

-- TRAINING_LOG
ALTER TABLE IF EXISTS public.training_log
    ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION NOT VALID;

ALTER TABLE public.training_log VALIDATE CONSTRAINT fk_users_id;

ALTER TABLE IF EXISTS public.training_log
    ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE NO ACTION NOT VALID;

ALTER TABLE public.training_log VALIDATE CONSTRAINT fk_training_id;

-- MODELS
-- training id
ALTER TABLE IF EXISTS public.models
    ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.models VALIDATE CONSTRAINT fk_training_id;

-- framework id
ALTER TABLE IF EXISTS public.models
    ADD CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.models VALIDATE CONSTRAINT fk_framework_id;

-- deployment id
ALTER TABLE IF EXISTS public.models
    ADD CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.models VALIDATE CONSTRAINT fk_deployment_id;

-- model type id
ALTER TABLE IF EXISTS public.models
    ADD CONSTRAINT fk_model_type_id FOREIGN KEY (model_type_id) REFERENCES public.model_type (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.models VALIDATE CONSTRAINT fk_model_type_id;


/*  NOTE DEPRECATED */
/* -- PRE-TRAINED MODELS
ALTER TABLE IF EXISTS public.pre_trained_models
 ADD CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.pre_trained_models VALIDATE CONSTRAINT fk_framework_id;

ALTER TABLE IF EXISTS public.pre_trained_models
 ADD CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.pre_trained_models VALIDATE CONSTRAINT fk_deployment_id;

-- USER_UPLOAD_MODELS table
ALTER TABLE IF EXISTS public.user_upload_models
 ADD CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.user_upload_models VALIDATE CONSTRAINT fk_framework_id;

ALTER TABLE IF EXISTS public.user_upload_models
 ADD CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.user_upload_models VALIDATE CONSTRAINT fk_deployment_id;
 */
-- PREDICTIONS
ALTER TABLE IF EXISTS public.predictions
    ADD CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES public.models (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.predictions VALIDATE CONSTRAINT fk_model_id;


ALTER TABLE IF EXISTS public.predictions
    ADD CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.predictions VALIDATE CONSTRAINT fk_task_id;

-- DATASET
ALTER TABLE IF EXISTS public.dataset
    ADD CONSTRAINT fk_filetype_id FOREIGN KEY (filetype_id) REFERENCES public.filetype (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.dataset VALIDATE CONSTRAINT fk_filetype_id;

--TASK
ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_dataset_id;

ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_project_id;

ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_annotation_id FOREIGN KEY (annotation_id) REFERENCES public.annotations (id) ON DELETE SET NULL SET NULL NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_annotation_id;

ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_prediction_id FOREIGN KEY (prediction_id) REFERENCES public.predictions (id) ON DELETE SET NULL SET NULL NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_prediction_id;

-- ANNOTATIONS
ALTER TABLE IF EXISTS public.annotations
    ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION NOT VALID;

ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_users_id;

-- ALTER TABLE IF EXISTS public.annotations
--     ADD CONSTRAINT fk_annotation_type_id FOREIGN KEY (annotation_type_id) REFERENCES public.annotation_type (id) ON DELETE NO ACTION NOT VALID;
ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_annotation_type_id;

ALTER TABLE IF EXISTS public.annotations
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_project_id;

ALTER TABLE IF EXISTS public.annotations
    ADD CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_task_id;

-- PROJECT_DATASET (Many-to-Many)
ALTER TABLE IF EXISTS public.project_dataset
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_dataset VALIDATE CONSTRAINT fk_project_id;

ALTER TABLE IF EXISTS public.project_dataset
    ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_dataset VALIDATE CONSTRAINT fk_dataset_id;

-- EDITOR
ALTER TABLE IF EXISTS public.editor
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.editor VALIDATE CONSTRAINT fk_project_id;

-- PROJECT_TRAINING (Many-to-Many)
ALTER TABLE IF EXISTS public.project_training
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_training VALIDATE CONSTRAINT fk_project_id;

ALTER TABLE IF EXISTS public.project_training
    ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_training VALIDATE CONSTRAINT fk_training_id;

-- TRAINING_DATASET (Many-to-Many)
ALTER TABLE IF EXISTS public.training_dataset
    ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.training_dataset VALIDATE CONSTRAINT fk_training_id;

ALTER TABLE IF EXISTS public.training_dataset
    ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.training_dataset VALIDATE CONSTRAINT fk_dataset_id;

