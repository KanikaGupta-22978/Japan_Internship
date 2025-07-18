class Admin::MasqueradesController < Devise::MasqueradesController
  protected

  def find_masqueradable_resource
  end

  def after_masquerade_path_for(resource)
    user_url(resource)
  end

  def after_back_masquerade_path_for(resource)
    root_url(resource)
  end
end