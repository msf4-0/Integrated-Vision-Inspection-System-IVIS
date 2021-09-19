CREATE DATABASE integrated_vision_inspection_system;

BEGIN;

--
-- Name: trigger_update_timestamp(); Type: FUNCTION; Schema: public; Owner:
--
CREATE OR REPLACE FUNCTION trigger_update_timestamp ()
    RETURNS TRIGGER
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

-- ACCOUNT_STATUS table ---------------------------
CREATE TABLE IF NOT EXISTS public.account_status (
    id integer NOT NULL GENERATED ALWAYS AS IDENTITY
    , name character varying(50) UNIQUE NOT NULL
    , PRIMARY KEY (id)
);

INSERT INTO public.account_status (
    name)
VALUES (
    'NEW')
, (
    'ACTIVE')
, (
    'LOCKED')
, (
    'LOGGED_IN')
, (
    'LOGGED_OUT');

-- ANNOTATIONS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.annotations (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , result jsonb[]
    , annotation_type_id integer
    , project_id bigint NOT NULL
    , users_id bigint NOT NULL
    , task_id bigint NOT NULL
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER annotations_update
    BEFORE UPDATE ON public.annotations
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- DATASET table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.dataset (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name text NOT NULL UNIQUE
    , description text
    , dataset_path text NOT NULL
    , dataset_size integer
    , filetype_id integer
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER dataset_update
    BEFORE UPDATE ON public.dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

--  DEPLOYMENT_TYPE table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.deployment_type (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name character varying(100) NOT NULL
    , template TEXT
    , PRIMARY KEY (id)
);

INSERT INTO public.deployment_type (
    name)
VALUES (
    'Image Classification')
, (
    'Object Detection with Bounding Boxes')
, (
    'Semantic Segmentation with Polygons');

-- EDITOR table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.editor (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name character varying(50) NOT NULL
    , editor_config text
    , labels jsonb[]
    , project_id bigint NOT NULL
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER editor_update
    BEFORE UPDATE ON public.editor
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- FILETYPE table ---------------------------
CREATE TABLE IF NOT EXISTS public.filetype (
    id integer NOT NULL GENERATED ALWAYS AS IDENTITY
    , name character varying(50) UNIQUE NOT NULL
    , PRIMARY KEY (id)
);

INSERT INTO public.filetype (
    name)
VALUES (
    'Image') ,
--jpeg,jpg,png
(
    'Video') ,
-- mp4,mpeg,webm*,ogg*
(
    'Audio') ,
--* wav, aiff, mp3, au, flac, m4a, ogg
(
    'Text') -- txt,csv,tsv,json*,html*
;

-- FRAMEWORK table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.framework (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name character varying(50) NOT NULL
    , link text
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER framework_update
    BEFORE UPDATE ON public.framework
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

INSERT INTO public.framework (
    name
    , link)
VALUES (
    'TensorFlow'
    , 'https://www.tensorflow.org/')
, (
    'PyTorch'
    , 'https://pytorch.org/')
, (
    'Scikit-learn'
    , 'https://scikit-learn.org/stable/')
, (
    'Caffe'
    , 'https://caffe.berkeleyvision.org/')
, (
    'MXNet'
    , 'https://mxnet.apache.org/')
, (
    'ONNX'
    , 'https://onnx.ai/');

-- MODEL_TYPE table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.model_type (
    id smallint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 32767
    CACHE 1)
    , name character varying(50) NOT NULL
    , PRIMARY KEY (id)
);

INSERT INTO public.model_type (
    name)
VALUES (
    'Pre-trained Models')
, (
    'Project Models')
, (
    'User Custom Deep Learning Model Upload');

-- MODELS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.models (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name text NOT NULL UNIQUE
    , description text
    , metrics jsonb
    ,
    /*to store evaluation metrics for model such as COCO mAP and Training Losses*/
    model_path text
    , model_type_id smallint
    , framework_id bigint
    , deployment_id integer
    , training_id bigint
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER models_update
    BEFORE UPDATE ON public.models
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- PREDICTIONS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.predictions (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , result jsonb[]
    , task_id bigint NOT NULL
    , model_id bigint
    , pre_trained_model_id bigint
    , score double precision
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER predictions_update
    BEFORE UPDATE ON public.predictions
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

--  PROJECT table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.project (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name text NOT NULL UNIQUE
    , description text
    , deployment_id integer
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER project_update
    BEFORE UPDATE ON public.project
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- PROJECT_DATASET table (Many-to-Many) --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.project_dataset (
    project_id bigint NOT NULL
    , dataset_id bigint NOT NULL
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (project_id , dataset_id)
);

CREATE TRIGGER project_dataset_update
    BEFORE UPDATE ON public.project_dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- PROJECT_TRAINING table (Many-to-Many) ---------------------------
CREATE TABLE IF NOT EXISTS public.project_training (
    project_id bigint NOT NULL
    , training_id bigint NOT NULL
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (project_id , training_id)
);

CREATE TRIGGER project_training_update
    BEFORE UPDATE ON public.project_training
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- CHECK (account_status IN ('NEW', 'ACTIVE', 'LOCKED', 'LOGGED_IN', 'LOGGED_OUT'))
--  ROLES table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.roles (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name character varying(50) NOT NULL
    , page_access_list text[]
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER roles_update
    BEFORE UPDATE ON public.roles
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- INSERT pre-defined values into ROLES
INSERT INTO public.roles (
    name
    , page_access_list)
VALUES (
    'Administrator'
    , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor' , 'Model Training' , 'Deployment' , 'User Management'])
, (
    'Developer 1 (Deployment)'
    , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor' , 'Model Training' , 'Deployment'])
, (
    'Developer 2 (Model Training)'
    , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor' , 'Model Training'])
, (
    'Annotator'
    , ARRAY['Login' , 'Project' , 'Dataset' , 'Editor']);

--  SESSION_LOG table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.session_log (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , users_id bigint NOT NULL
    , login_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , logout_at timestamp with time zone NOT NULL
    , PRIMARY KEY (id)
);

--  TASK table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.task (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name text NOT NULL
    , dataset_id bigint NOT NULL
    , project_id bigint
    , annotation_id bigint
    , prediction_id bigint
    , is_labelled boolean NOT NULL DEFAULT FALSE
    , skipped boolean NOT NULL DEFAULT FALSE
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (id)
);

CREATE TRIGGER task_update
    BEFORE UPDATE ON public.task
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

--  TRAINING table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.training (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , name text NOT NULL UNIQUE
    , description text
    , training_model_id bigint
    ,
    /* FK to 'model' table for model trained using current configuration */
    attached_model_id bigint
    ,
    /* FK to 'models' table for models attached to current training */
    training_param jsonb
    , augmentation jsonb
    , partition_size real
    , is_started boolean NOT NULL DEFAULT FALSE
    , progress jsonb
    ,
    /* stores progress tracking of model training such as checkpoint and previous training steps*/
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , project_id bigint NOT NULL
    , PRIMARY KEY (id)
);

CREATE TRIGGER training_update
    BEFORE UPDATE ON public.training
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- TRAINING_DATASET table (Many-to-Many) ---------------------------
CREATE TABLE IF NOT EXISTS public.training_dataset (
    training_id bigint NOT NULL
    , dataset_id bigint NOT NULL
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , PRIMARY KEY (training_id , dataset_id)
);

CREATE TRIGGER training_dataset_update
    BEFORE UPDATE ON public.training_dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- TRAINING_LOG table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.training_log (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , users_id bigint NOT NULL
    , training_id bigint NOT NULL
    , start_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP ,
    --update using Python
    end_at timestamp with time zone NOT NULL
    , PRIMARY KEY (id)
);

-- USERS table ------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1)
    , emp_id text UNIQUE
    , username text NOT NULL UNIQUE
    , first_name text
    , last_name text
    , email text
    , department text
    , position text
    , psd text NOT NULL
    , roles_id integer NOT NULL
    , status_id integer NOT NULL DEFAULT 1
    , created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP
    , last_activity timestamp with time zone
    , PRIMARY KEY (id)
);

CREATE TRIGGER users_update
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- Default Admin Account
/* 
emp_id:0
username:admin
password:admin
Role:Admininstrator */
INSERT INTO public.users (
    emp_id
    , username
    , first_name
    , last_name
    , email
    , department
    , position
    , psd
    , roles_id)
VALUES (
    '0'
    , 'admin'
    , NULL
    , NULL
    , NULL
    , NULL
    , NULL
    , 'admin'
    , (
        SELECT
            id
        FROM
            public.roles
        WHERE
            name = 'Administrator'));

-----------------------------------Foreign Keys Constraint---------------------------------------------------
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

ALTER TABLE public.session_log VALIDATE CONSTRAINT fk_users_id;

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
-- dataset id
ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_dataset_id;

-- project id
ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_project_id;

-- annotation id
ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_annotation_id FOREIGN KEY (annotation_id) REFERENCES public.annotations (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_annotation_id;

-- prediction id
ALTER TABLE IF EXISTS public.task
    ADD CONSTRAINT fk_prediction_id FOREIGN KEY (prediction_id) REFERENCES public.predictions (id) ON DELETE SET NULL NOT VALID;

ALTER TABLE public.task VALIDATE CONSTRAINT fk_prediction_id;

-- ANNOTATIONS
-- users id
ALTER TABLE IF EXISTS public.annotations
    ADD CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION NOT VALID;

ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_users_id;

-- project id
ALTER TABLE IF EXISTS public.annotations
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_project_id;

-- task id
ALTER TABLE IF EXISTS public.annotations
    ADD CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.annotations VALIDATE CONSTRAINT fk_task_id;

-- PROJECT_DATASET (Many-to-Many)
-- project id
ALTER TABLE IF EXISTS public.project_dataset
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_dataset VALIDATE CONSTRAINT fk_project_id;

-- dataset id
ALTER TABLE IF EXISTS public.project_dataset
    ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_dataset VALIDATE CONSTRAINT fk_dataset_id;

-- EDITOR
-- project id
ALTER TABLE IF EXISTS public.editor
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.editor VALIDATE CONSTRAINT fk_project_id;

-- PROJECT_TRAINING (Many-to-Many)
-- project id
ALTER TABLE IF EXISTS public.project_training
    ADD CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_training VALIDATE CONSTRAINT fk_project_id;

-- training id
ALTER TABLE IF EXISTS public.project_training
    ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.project_training VALIDATE CONSTRAINT fk_training_id;

-- TRAINING_DATASET (Many-to-Many)
-- training id
ALTER TABLE IF EXISTS public.training_dataset
    ADD CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.training_dataset VALIDATE CONSTRAINT fk_training_id;

-- dataset id
ALTER TABLE IF EXISTS public.training_dataset
    ADD CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE NOT VALID;

ALTER TABLE public.training_dataset VALIDATE CONSTRAINT fk_dataset_id;

END;

