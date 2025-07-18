class ApiController < ActionController::Base
  include Response
  include ExceptionHandler

	skip_before_action :verify_authenticity_token
end