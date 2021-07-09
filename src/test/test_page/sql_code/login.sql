-- >>>> New Session Log at Log-IN >>>>
INSERT INTO session_log (user_id)
VALUES (%s)
RETURNING id;
-- this state would include id, user_id, login_at

-- >>>> Update Session Log at Log-OUT >>>>
UPDATE session_log 
SET logout_at = %s --here substitute current datetime values
WHERE id = %s -- Current session_id
RETURNING *;
-- this state would include id, user_id, login_at,logout_at (COMPLETE)


-- >>>> Update Account Status
UPDATE user 
SET status = %s -- substitute account status
WHERE id = %s; -- user_id


