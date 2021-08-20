-- Title: Integrated Vision Inspection System
-- Date:10/7/2021
-- Author: Chu Zhen Hao
--  Organisation: Malaysian Smart Factory 4.0 Team at Selangor Human Resource Development Centre (SHRDC)
BEGIN;

CREATE ROLE integrated_vision_inspection_system WITH superuser createdb createrole noinherit encrypted PASSWORD 'integrated_vision_inspection_system';

CREATE TABLESPACE IF NOT EXISTS integrated_vision_inspection_system OWNER integrated_vision_inspection_system location '$HOME/.local/share/integrated_vision_inspection_system';

-- CREATE DATABASE IF NOT EXISTS integrated_vision_inspection_system OWNER integrated_vision_inspection_system TABLESPACE integrated_vision_inspection_system;
-- CREATE DATABASE IF NOT EXISTS eye OWNER shrdc TABLESPACE shrdcdb;
CREATE DATABASE IF NOT EXISTS eye OWNER shrdc TABLESPACE image_labelling;

-- Trigger Function for Update Time
CREATE OR REPLACE FUNCTION trigger_update_timestamp ()
    RETURNS TRIGGER
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

ALTER FUNCTION public.trigger_update_timestamp () OWNER TO shrdc;

-- USERS table ------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    emp_id text UNIQUE,
    username text NOT NULL UNIQUE,
    first_name text,
    last_name text,
    email text,
    department text,
    position text,
    psd text NOT NULL,
    roles_id integer NOT NULL,
    status_id integer NOT NULL DEFAULT 1,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_activity timestamp with time zone,
    PRIMARY KEY (id),
    CONSTRAINT fk_roles_id FOREIGN KEY (roles_id) REFERENCES public.roles (id) ON DELETE SET NULL,
    CONSTRAINT fk_status_id FOREIGN KEY (status_id) REFERENCES public.account_status (id) ON DELETE SET NULL ON UPDATE CASCADE)
TABLESPACE image_labelling;

CREATE TRIGGER users_update
    BEFORE UPDATE ON public.users
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.users OWNER TO shrdc;

INSERT INTO public.users (
    emp_id,
    username,
    first_name,
    last_name,
    email,
    department,
    position,
    psd,
    roles_id)
VALUES (
    'shrdc1',
    'chuzhenhao_shrdc',
    'Zhen Hao',
    'Chu',
    'chuzhenhao@shrdc.com',
    'Engineering',
    'Intern',
    'shrdc',
    (
        SELECT
            id
        FROM
            public.roles
        WHERE
            name = 'Administrator'));

-- ALTER TABLE public.users
--     DROP CONSTRAINT users_account_status_check;
-- CHECK (account_status IN ('NEW', 'ACTIVE', 'LOCKED', 'LOGGED_IN', 'LOGGED_OUT'))
--  ROW table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.roles (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(50) NOT NULL,
    page_access_list text[],
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.roles OWNER TO shrdc;

CREATE TRIGGER roles_update
    BEFORE UPDATE ON public.roles
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

-- INSERT pre-defined values into ROLES
INSERT INTO public.roles (
    name,
    page_access_list)
VALUES (
    'Administrator',
    ARRAY['Login', 'Project', 'Dataset', 'Editor', 'Model Training', 'Deployment', 'User Management']),
(
    'Developer 1 (Deployment)',
    ARRAY['Login', 'Project', 'Dataset', 'Editor', 'Model Training', 'Deployment']),
(
    'Developer 2 (Model Training)',
    ARRAY['Login', 'Project', 'Dataset', 'Editor', 'Model Training']),
(
    'Annotator',
    ARRAY['Login', 'Project', 'Dataset', 'Editor']);

--  SESSION_LOG table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.session_log (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    users_id bigint NOT NULL,
    login_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    logout_at timestamp with time zone NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION)
TABLESPACE image_labelling;

ALTER TABLE public.session_log OWNER TO shrdc;

--  PROJECT table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.project (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    description text,
    project_path text NOT NULL,
    deployment_id integer,
    training_id bigint,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL,
    CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE SET NULL)
TABLESPACE image_labelling;

CREATE TRIGGER project_update
    BEFORE UPDATE ON public.project
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.project OWNER TO shrdc;

--  TRAINING table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.training (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    description text,
    training_param jsonb[],
    augmentation jsonb[],
    model_id bigint,
    partition_size real,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    project_id bigint NOT NULL,
    pre_trained_model_id bigint,
    framework_id bigint,
    PRIMARY KEY (id),
    CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES public.models (id) ON DELETE SET NULL,
    CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE,
    CONSTRAINT fk_pre_trained_model_id FOREIGN KEY (pre_trained_model_id) REFERENCES public.pre_trained_models (id) ON DELETE SET NULL,
    CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL)
TABLESPACE image_labelling;

CREATE TRIGGER training_update
    BEFORE UPDATE ON public.training
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.training OWNER TO shrdc;

-- TRAINING_LOG table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.training_log (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    users_id bigint NOT NULL,
    training_id bigint NOT NULL,
    start_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP, --update using Python
    end_at timestamp with time zone NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION,
    CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE NO ACTION)
TABLESPACE image_labelling;

ALTER TABLE public.training_log OWNER TO shrdc;

-- PRE-TRAINED MODELS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.pre_trained_models (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (CYCLE INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL,
    model_path text,
    framework_id bigint,
    deployment_id integer,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL,
    CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL)
TABLESPACE image_labelling;

CREATE TRIGGER pre_trained_models_update
    BEFORE UPDATE ON public.pre_trained_models
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.pre_trained_models OWNER TO shrdc;

-- MODELS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.models (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    model_path text,
    training_id bigint,
    framework_id bigint,
    deployment_id integer,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE SET NULL,
    CONSTRAINT fk_framework_id FOREIGN KEY (framework_id) REFERENCES public.framework (id) ON DELETE SET NULL,
    CONSTRAINT fk_deployment_id FOREIGN KEY (deployment_id) REFERENCES public.deployment_type (id) ON DELETE SET NULL)
TABLESPACE image_labelling;

CREATE TRIGGER models_update
    BEFORE UPDATE ON public.models
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.models OWNER TO shrdc;

-- PREDICTIONS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.predictions (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    result jsonb[],
    task_id bigint NOT NULL,
    model_id bigint,
    pre_trained_model_id bigint,
    score double precision,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES public.models (id) ON DELETE SET NULL,
    CONSTRAINT fk_pre_trained_model_id FOREIGN KEY (pre_trained_model_id) REFERENCES public.pre_trained_models (id) ON DELETE SET NULL,
    CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE)
TABLESPACE image_labelling;

CREATE TRIGGER predictions_update
    BEFORE UPDATE ON public.predictions
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.predictions OWNER TO shrdc;

-- DATASET table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.dataset (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL UNIQUE,
    description text,
    dataset_path text NOT NULL,
    dataset_size integer,
    filetype_id integer,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_filetype_id FOREIGN KEY (filetype_id) REFERENCES public.filetype (id) ON DELETE SET NULL)
TABLESPACE image_labelling;

CREATE TRIGGER dataset_update
    BEFORE UPDATE ON public.dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.dataset OWNER TO shrdc;

--  DEPLOYMENT_TYPE table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.deployment_type (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(100) NOT NULL,
    template text,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.deployment_type OWNER TO shrdc;

--  TASK table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.task (
    id bigint NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name text NOT NULL,
    dataset_id bigint NOT NULL,
    project_id bigint,
    annotation_id bigint,
    prediction_id bigint,
    is_labelled boolean NOT NULL DEFAULT FALSE,
    skipped boolean NOT NULL DEFAULT FALSE,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE, --KIV
    CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE,
    CONSTRAINT fk_annotation_id FOREIGN KEY (annotation_id) REFERENCES public.annotations (id) ON DELETE SET NULL,
    CONSTRAINT fk_prediction_id FOREIGN KEY (prediction_id) REFERENCES public.predictions (id) ON DELETE SET NULL)
TABLESPACE image_labelling;

CREATE TRIGGER task_update
    BEFORE UPDATE ON public.task
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.task OWNER TO shrdc;

-- ANNOTATIONS table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.annotations (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    result jsonb[],
    annotation_type_id integer,
    project_id bigint NOT NULL,
    users_id bigint NOT NULL,
    task_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_users_id FOREIGN KEY (users_id) REFERENCES public.users (id) ON DELETE NO ACTION,
    CONSTRAINT fk_annotation_type_id FOREIGN KEY (annotation_type_id) REFERENCES public.annotation_type (id) ON DELETE NO ACTION,
    CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE,
    CONSTRAINT fk_task_id FOREIGN KEY (task_id) REFERENCES public.task (id) ON DELETE CASCADE)
TABLESPACE image_labelling;

CREATE TRIGGER annotations_update
    BEFORE UPDATE ON public.annotations
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.annotations OWNER TO shrdc;

-- ANNOTATION_TYPE table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.annotation_type (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(100) NOT NULL,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.annotation_type OWNER TO shrdc;

-- PROJECT_DATASET table (Many-to-Many) --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.project_dataset (
    project_id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, dataset_id),
    CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE,
    CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE)
TABLESPACE image_labelling;

ALTER TABLE public.project_dataset OWNER TO shrdc;

-- EDITOR table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.editor (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(50) NOT NULL,
    editor_config text,
    labels jsonb[],
    project_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE)
TABLESPACE image_labelling;

CREATE TRIGGER editor_update
    BEFORE UPDATE ON public.editor
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.editor OWNER TO shrdc;

-- FRAMEWORK table --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.framework (
    id bigint NOT NULL GENERATED ALWAYS AS IDENTITY (INCREMENT 1 START 1
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1),
    name character varying(50) NOT NULL,
    link text,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

CREATE TRIGGER framework_update
    BEFORE UPDATE ON public.framework
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.framework OWNER TO shrdc;

INSERT INTO public.framework (
    name,
    link)
VALUES (
    'TensorFlow',
    'https://www.tensorflow.org/'),
(
    'PyTorch',
    'https://pytorch.org/'),
(
    'Scikit-learn',
    'https://scikit-learn.org/stable/'),
(
    'Caffe',
    'https://caffe.berkeleyvision.org/'),
(
    'MXNet',
    'https://mxnet.apache.org/'),
(
    'ONNX',
    'https://onnx.ai/');

-- PROJECT_TRAINING table (Many-to-Many) ---------------------------
CREATE TABLE IF NOT EXISTS public.project_training (
    project_id bigint NOT NULL,
    training_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (project_id, training_id),
    CONSTRAINT fk_project_id FOREIGN KEY (project_id) REFERENCES public.project (id) ON DELETE CASCADE,
    CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE)
TABLESPACE image_labelling;

CREATE TRIGGER project_training_update
    BEFORE UPDATE ON public.project_training
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.project_training OWNER TO shrdc;

-- TRAINING_DATASET table (Many-to-Many) ---------------------------
CREATE TABLE IF NOT EXISTS public.training_dataset (
    training_id bigint NOT NULL,
    dataset_id bigint NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (training_id, dataset_id),
    CONSTRAINT fk_training_id FOREIGN KEY (training_id) REFERENCES public.training (id) ON DELETE CASCADE,
    CONSTRAINT fk_dataset_id FOREIGN KEY (dataset_id) REFERENCES public.dataset (id) ON DELETE CASCADE)
TABLESPACE image_labelling;

CREATE TRIGGER training_dataset_update
    BEFORE UPDATE ON public.training_dataset
    FOR EACH ROW
    EXECUTE PROCEDURE trigger_update_timestamp ();

ALTER TABLE public.training_dataset OWNER TO shrdc;

-- ACCOUNT_STATUS table ---------------------------
CREATE TABLE IF NOT EXISTS public.account_status (
    id integer NOT NULL GENERATED ALWAYS AS IDENTITY,
    name character varying(50) UNIQUE NOT NULL,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.account_status OWNER TO shrdc;

INSERT INTO public.account_status (
    name)
VALUES (
    'NEW'),
(
    'ACTIVE'),
(
    'LOCKED'),
(
    'LOGGED_IN'),
(
    'LOGGED_OUT');

-- FILETYPE table ---------------------------
CREATE TABLE IF NOT EXISTS public.filetype (
    id integer NOT NULL GENERATED ALWAYS AS IDENTITY,
    name character varying(50) UNIQUE NOT NULL,
    PRIMARY KEY (id))
TABLESPACE image_labelling;

ALTER TABLE public.filetype OWNER TO shrdc;

INSERT INTO public.filetype (
    name)
VALUES (
    'Image'), --jpeg,jpg,png
(
    'Video'), -- mp4,mpeg,webm*,ogg*
(
    'Audio'), --* wav, aiff, mp3, au, flac, m4a, ogg
(
    'Text') -- txt,csv,tsv,json*,html*
;

END;

