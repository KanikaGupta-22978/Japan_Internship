ActiveAdmin.register User do
  permit_params :uid, :user_access_token, :user_access_token_secret, :remarks

  actions :all, except: [:destroy, :new, :create, :show]
    # :edit, :update

  index do
    selectable_column
    id_column
    column :remarks
    column :uid
    column :provider
    column :user_access_token
    column :user_access_token_secret
    column :created_at
    column :updated_at
    column "Number of Summaries" do |user|
      strong "Heart Rate: "
      span user.dailies.count
      br
      strong "Sleep: "
      span user.sleeps.count
      br
      strong "Stress: "
      span user.stresses.count
      br
      strong "Epoch: "
      span user.epochs.count
    end
    column "Days with Summaries" do |user|
      strong "Heart Rate: "
      span user.dailies.pluck("data->>'calendarDate'").min
      br
      span user.dailies.pluck("data->>'calendarDate'").max
      br
      br

      strong "Sleep: "
      span user.sleeps.pluck("data->>'calendarDate'").min
      br
      span user.sleeps.pluck("data->>'calendarDate'").max
      br
      br

      strong "Stress: "
      span user.stresses.pluck("data->>'calendarDate'").min
      br
      span user.stresses.pluck("data->>'calendarDate'").max
    end
    actions
    column () do |resource|
      link_to "Login in as #{resource.id}", masquerade_path(resource)
    end
  end

  filter :created_at
  filter :updated_at
  filter :remarks

  form do |f|
    f.inputs 'User Details' do
      f.input :remarks
    end
    f.actions
  end
end
