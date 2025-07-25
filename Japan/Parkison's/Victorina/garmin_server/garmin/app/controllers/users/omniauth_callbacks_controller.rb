class Users::OmniauthCallbacksController < Devise::OmniauthCallbacksController
	def garmin
    # You need to implement the method below in your model (e.g. app/models/user.rb)
    @user = User.from_omniauth(request.env["omniauth.auth"])

    if @user.persisted?
      sign_in_and_redirect @user, event: :authentication #this will throw if @user is not activated
      set_flash_message(:notice, :success, kind: "Garmin") if is_navigational_format?
    else
      redirect_to root_path
    end
	end

	def failure
    redirect_to root_path
  end
end