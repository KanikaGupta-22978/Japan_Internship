class AddRemarksToUsers < ActiveRecord::Migration[5.2]
  def change
    add_column :users, :remarks, :text
  end
end
